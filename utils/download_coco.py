import os
import requests
from zipfile import ZipFile
from urllib.request import urlopen
from tqdm import tqdm
import shutil

DATASET_ROOT = "coco/"
TRAIN2017_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
TRAINVAL2017_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
block_size = 1024 # 1 KB

def download_coco_2017():

    if os.path.exists(DATASET_ROOT) == False:
        print("Could not find dataset root dir. Creating...")
        os.mkdir(DATASET_ROOT)
    skip_download = False
    print("Downloading COCO 2017 train images...")
    if os.path.exists(os.path.join(DATASET_ROOT, 'images/')) == False:
        os.mkdir(os.path.join(DATASET_ROOT, "images"))
    if os.path.isfile(os.path.join(DATASET_ROOT, 'images', 'train2017.1.zip')):
        print("COCO train2017 zip file found, skipping download...")
        skip_download = True

    # Download COCO dataset train vals

    if skip_download == False:        
        site = urlopen(TRAIN2017_IMAGES_URL)
        meta = site.info()
        total_size_in_bytes = int(meta['Content-Length'])
        # Streaming so we know how the download's going
        response = requests.get(TRAIN2017_IMAGES_URL, stream=True)
        with open(os.path.join(DATASET_ROOT, 'images/', 'train2017.1.zip'), 'wb+') as f:
            for data in (pbar := tqdm(response.iter_content(block_size), total=total_size_in_bytes, unit_scale=True)):
                pbar.update(len(data))
                f.write(data)
        pbar.close()

    # Unzip
    # print("Unzipping, might take a while...")
    # with ZipFile(os.path.join(DATASET_ROOT, 'images/', 'train2017.1.zip'), 'r') as zip:
    #     zip.extractall(os.path.join(DATASET_ROOT, 'images/'))

    if os.path.exists(os.path.join(DATASET_ROOT, "annotations/")) == False:
        os.mkdir(os.path.join(DATASET_ROOT, "annotations/"))

    site = urlopen(TRAINVAL2017_ANNOTATIONS_URL)
    meta = site.info()
    total_size_in_bytes = int(meta['Content-Length'])
    response = requests.get(TRAINVAL2017_ANNOTATIONS_URL, stream=True)
    with open(os.path.join(DATASET_ROOT, 'annotations/', 'ann2017.zip'), 'wb+') as f:
        for data in (pbar := tqdm(response.iter_content(block_size), total=total_size_in_bytes, unit_scale=True)):
            pbar.update(len(data))
            f.write(data)
    pbar.close()

    if os.path.exists(os.path.join(DATASET_ROOT, "annotations/", "ann2017")) == False:
        os.mkdir(os.path.join(DATASET_ROOT, "annotations/", "ann2017"))

    print("Unzipping, might take a while...")
    with ZipFile(os.path.join(DATASET_ROOT, 'annotations/', 'ann2017.zip'), 'r') as zip:
        for file in zip.namelist():
            if file.startswith('annotations/'):
                filename = file.split('/')[-1]

                source = zip.open(file)
                target = open(os.path.join(DATASET_ROOT, 'annotations/', 'ann2017/', filename), 'wb+')
                with source, target:
                    shutil.copyfileobj(source, target)

if __name__ == "__main__":
    download_coco_2017()