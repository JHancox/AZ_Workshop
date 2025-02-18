# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections.abc
import os
import shutil
import time

import gdown
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import nvtx

@nvtx.annotate("train_epoch", color="blue")
def train_epoch(model, loader, optimizer, scaler, epoch, epochs):
    """One train epoch over the dataset"""

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    for idx, batch_data in enumerate(loader):
        with nvtx.annotate("Start batch", color="green"):
            with nvtx.annotate("get data", color="yellow"):
                data = batch_data["image"].as_subclass(torch.Tensor).cuda(0)
                target = batch_data["label"].as_subclass(torch.Tensor).cuda(0)
                
            with nvtx.annotate("zero optimizer", color="cyan"):
                optimizer.zero_grad(set_to_none=True)

            with nvtx.annotate("do inference", color="red"):    
                with autocast(enabled=True):
                    logits = model(data)
                    loss = criterion(logits, target)

            with nvtx.annotate("back_prop", color="grey"): 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            with nvtx.annotate("metrics", color="purple"):
                acc = (logits.sigmoid().sum(1).detach().round() == target.sum(1).round()).float().mean()

                run_loss.append(loss)
                run_acc.append(acc)

                loss = run_loss.aggregate()
                acc = run_acc.aggregate()

                print(
                    "Epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                    "loss: {:.4f}".format(loss),
                    "acc: {:.4f}".format(acc),
                    "time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()

    return loss, acc

@nvtx.annotate("val_epoch", color="blue")
def val_epoch(model, loader, epoch, epochs, max_tiles=None):
    """One validation epoch over the dataset"""

    model.eval()

    model2 = model
    calc_head = model2.calc_head

    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    PREDS = Cumulative()
    TARGETS = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).cuda(0)
            target = batch_data["label"].as_subclass(torch.Tensor).cuda(0)

            with autocast(enabled=True):
                if max_tiles is not None and data.shape[1] > max_tiles:
                    # During validation, we want to use all instances/patches
                    # and if its number is very big, we may run out of GPU memory
                    # in this case, we first iteratively go over subsets of patches to calculate backbone features
                    # and at the very end calculate the classification output

                    logits = []
                    logits2 = []

                    for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                        data_slice = data[:, i * max_tiles : (i + 1) * max_tiles]
                        logits_slice = model(data_slice, no_head=True)
                        logits.append(logits_slice)
                        
                    logits = torch.cat(logits, dim=1)
                    logits = calc_head(logits)

                else:
                    # if number of instances is not big, we can run inference directly
                    logits = model(data)

                loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            target = target.sum(1).round()
            acc = (pred == target).float().mean()

            run_loss.append(loss)
            run_acc.append(acc)
            loss = run_loss.aggregate()
            acc = run_acc.aggregate()

            PREDS.extend(pred)
            TARGETS.extend(target)

            print(
                "Val epoch {}/{} {}/{}".format(epoch, epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss),
                "acc: {:.4f}".format(acc),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        PREDS = PREDS.get_buffer().cpu().numpy()
        TARGETS = TARGETS.get_buffer().cpu().numpy()
        qwk = cohen_kappa_score(PREDS.astype(np.float64), TARGETS.astype(np.float64), weights="quadratic")

    return loss, acc, qwk

@nvtx.annotate("LabelEncodeIntegerGraded", color="blue")
class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        num_classes: int,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d

@nvtx.annotate("list_data_collate", color="blue")
def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        # print(f"{i} = {item['image'].shape=} >> {item['image'].keys=}")
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        # data["patch_location"] = torch.stack([ix["patch_location"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def main_worker(args):

    torch.cuda.set_device(0)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    print("Batch size is:", args.batch_size, "epochs", args.epochs)

    #############
    # Create MONAI dataset
    training_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="training",
        base_dir=args.data_root,
    )
    validation_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="validation",
        base_dir=args.data_root,
    )

    if args.quick:  # for debugging on a small subset
        training_list = training_list[:16]
        validation_list = validation_list[:16]

    train_transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True),
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            RandGridPatchd(
                keys=["image"],
                patch_size=(args.tile_size, args.tile_size),
                num_patches=args.tile_count,
                sort_fn="min",
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(0), a_max=np.float32(255)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    valid_transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True),
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            GridPatchd(
                keys=["image"],
                patch_size=(args.tile_size, args.tile_size),
                threshold=0.999 * 3 * 255 * args.tile_size * args.tile_size,
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(0), a_max=np.float32(255)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    dataset_train = Dataset(data=training_list, transform=train_transform)
    dataset_valid = Dataset(data=validation_list, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    print("Dataset training:", len(dataset_train), "validation:", len(dataset_valid))

    model = milmodel.MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode)

    best_acc = 0
    start_epoch = 0

    model.cuda(0)
    params = model.parameters()
    m = model 
    params = [
        {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
        {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
    ]

    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.0

    scaler = GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, n_epochs):

        print(time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, epochs=n_epochs)

        print(
            "Final training  {}/{}".format(epoch, n_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "acc: {:.4f}".format(train_acc),
            "time {:.2f}s".format(time.time() - epoch_time),
        )


        b_new_best = False
        val_acc = 0
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=epoch, epochs=n_epochs, max_tiles=args.tile_count)

            print(
                "Final validation  {}/{}".format(epoch, n_epochs - 1),
                "loss: {:.4f}".format(val_loss),
                "acc: {:.4f}".format(val_acc),
                "qwk: {:.4f}".format(qwk),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
               
            val_acc = qwk

            if val_acc > val_acc_max:
                print("qwk ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                val_acc_max = val_acc
                print("New best model!")

        scheduler.step()

    print("ALL DONE")


def parse_args():
    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
    parser.add_argument(
        "--data_root", default="/PandaChallenge2020/train_images/", help="path to root folder of images"
    )
    parser.add_argument("--dataset_json", default=None, type=str, help="path to dataset json file")

    parser.add_argument("--num_classes", default=5, type=int, help="number of output classes")
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument(
        "--tile_count", default=44, type=int, help="number of patches (instances) to extract from WSI image"
    )
    parser.add_argument("--tile_size", default=256, type=int, help="size of square patch (instance) in pixels")

    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run only inference on the validation set, must specify the checkpoint argument",
    )

    parser.add_argument("--logdir", default=None, help="path to log directory to store Tensorboard logs")

    parser.add_argument("--epochs", "--max_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size, the number of WSI images per gpu")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument(
        "--val_every",
        "--val_interval",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument("--workers", default=2, type=int, help="number of workers for data loading")

    parser.add_argument("--quick", action="store_true", help="use a small subset of data for debugging")

    args = parser.parse_args()

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_json is None:
        # download default json datalist
        resource = "https://drive.google.com/uc?id=1L6PtKBlHHyUgTE4rVhRuOLTQKgD4tBRK"
        dst = "./datalist_panda_0.json"
        if not os.path.exists(dst):
            gdown.download(resource, dst, quiet=False)
        args.dataset_json = dst

    main_worker(args)
