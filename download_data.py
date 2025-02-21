import os
import requests

def download_and_extract(url, zip_dest_path):

    if not os.path.exists(zip_dest_path):
        print(f"Downloading dataset from {url} to {zip_dest_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        

script_dir = os.path.dirname(os.path.abspath(__file__))

zip_file_path = os.path.join(script_dir , "TNBC_NucleiSegmentation.zip")
download_url = "https://zenodo.org/record/3552674/files/TNBC_and_Brain_dataset.zip?download=1"
download_and_extract(download_url, zip_file_path)

zip_file_path = os.path.join(script_dir , "LyNSeC.zip")
download_url = "https://zenodo.org/record/8065174/files/lynsec.zip?download=1"
download_and_extract(download_url, zip_file_path)

zip_file_path = os.path.join(script_dir , "IHC_TMA_dataset.zip")
download_url = "https://zenodo.org/record/7647846/files/IHC_TMA_dataset.zip?download=1"
download_and_extract(download_url, zip_file_path)

zip_file_path = os.path.join(script_dir , "MoNuSeg.zip")
download_url = "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/monuseg-2018.zip"
download_and_extract(download_url, zip_file_path)

pannuke_dir = "/datasets/AZ_Workshop/data/PanNuke"

for fold in ["1", "2"]:
    zip_file_path = os.path.join(script_dir, "PanNuke_fold_{}.zip".format(fold))
    download_url = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{}.zip".format(fold)
    download_and_extract(download_url, zip_file_path)

print("Downloads completed.")
