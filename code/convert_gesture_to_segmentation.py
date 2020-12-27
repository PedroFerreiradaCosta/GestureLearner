"""


"""
import cv2
import numpy as np
import six.moves.urllib as urllib
import os
import zipfile
import scipy.io as sio

# https://drive.upm.es/index.php/s/efpozEx0D9tQhQa/download

#
# 1. Get image path
# 2. Transform images to torch tensors
# 3. Segment them using model
# img, _ = dataset_test[0]
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])
# 4. Save mask as Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
#








# rename image files so we dont have overlapping names
def rename_files(image_dir, rename_file):
    if rename_file:
        print("Renaming files")
        loop_index = 0
        for root, dirs, filenames in os.walk(image_dir):
            for dir in dirs:
                for f in os.listdir(image_dir + dir):
                    if (dir not in f):
                        if(f.split(".")[1] == "jpg"):
                            loop_index += 1
                            os.rename(image_dir + dir +
                                      "/" + f, image_dir + dir +
                                      "/" + dir + "_" + f)
                    else:
                        break
    generate_derivatives("../data/source/egohands/_LABELLED_SAMPLES/")

def extract_folder(dataset_path):
    print("Leap dataset already downloaded.")
    rename_file=False
    if not os.path.exists("../data/source/leap"):
        rename_file = True
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("../data/source/leap")
        print("> Extraction complete")
        zip_ref.close()
    rename_files("../data/source/leap_database/", rename_file)

def download_egohands_dataset(dataset_url, dataset_path):
    is_downloaded = os.path.exists(dataset_path)
    if not is_downloaded:
        print(
            "> downloading egohands dataset. This may take a while (1.3GB, say 3-5mins). Coffee break?")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, dataset_path)
        print("> download complete")
        extract_folder(dataset_path)
    else:
        extract_folder(dataset_path)


LEAP_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
LEAP_FILE = "../data/source/leap_database.rar"


download_egohands_dataset(LEAP_DATASET_URL, LEAP_FILE)
