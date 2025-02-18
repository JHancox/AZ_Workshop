import os
import json
import numpy as np
from sklearn.model_selection import KFold

def cellpose_json_file(dataset_dir, json_f_path):
    # This function takes in the directory of cellpose extracted dataset as input and
    # creates a json list with 5 folds. Separate testing set is recorded in the json list.
    # Please note that there are some hard-coded directory names as per the original dataset.
    # At the time of creation, the cellpose dataset had 'train.zip' and 'test.zip' that
    # extracted as 'train' and 'test' directories

    # The directory containing your files
    json_save_path = os.path.normpath(json_f_path)
    directory = os.path.join(dataset_dir, 'train')
    test_directory = os.path.join(dataset_dir, 'test')

    # List to hold all image-mask pairs
    data_pairs = []
    test_data_pairs = []
    all_data = {}

    # Scan the directory for image files and create pairs
    for filename in os.listdir(directory):
        if filename.endswith("_img.png"):
            # Construct the corresponding mask filename
            mask_filename = filename.replace("_img.png", "_masks.png")

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.normpath(os.path.join(directory, mask_filename))):
                # Add the pair to the list
                data_pairs.append({
                    "image": os.path.join('cellpose_dataset', 'train', filename),
                    "label": os.path.join('cellpose_dataset', 'train', mask_filename)
                })

    # Convert data_pairs to a numpy array for easy indexing by KFold
    data_pairs_array = np.array(data_pairs)

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Assign fold numbers
    for fold, (train_index, val_index) in enumerate(kf.split(data_pairs_array)):
        for idx in val_index:
            data_pairs_array[idx]['fold'] = fold

    # Convert the array back to a list and sort by fold
    sorted_data_pairs = sorted(data_pairs_array.tolist(), key=lambda x: x['fold'])

    print(sorted_data_pairs)

    # Scan the directory for image files and create pairs
    for filename in os.listdir(test_directory):
        if filename.endswith("_img.png"):
            # Construct the corresponding mask filename
            mask_filename = filename.replace("_img.png", "_masks.png")

            # Check if the corresponding mask file exists
            if os.path.exists(os.path.join(directory, mask_filename)):
                # Add the pair to the list
                test_data_pairs.append({
                    "image": os.path.join('cellpose_dataset', 'test', filename),
                    "label": os.path.join('cellpose_dataset', 'test', mask_filename)
                })

    all_data['training'] = sorted_data_pairs
    all_data['testing'] = test_data_pairs

    with open(json_save_path, 'w') as j_file:
        json.dump(all_data, j_file, indent=4)
    j_file.close()

def main():
    data_root = os.path.normpath('/define/the/data_path')
    save_path = os.path.normpath('/define/the/save_path.json')
    cellpose_json_file(dataset_dir=data_root, json_f_path=save_path)

if __name__=="__main__":
    main()