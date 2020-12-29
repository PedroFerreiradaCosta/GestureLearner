"""
Code extracted and adapted from `https://github.com/molyswu/hand_detection/blob/temp/hand_detection/egohands_dataset_clean.py`

Used to clean egoHands dataset.
Generates 2 image folders:
    * '../data/images/ - copies images from individual folders in source ds to images
    * '../data/masks/ - creates masks  of hands with name respective of image its referring to (up to 4 masks)


"""

import cv2
import numpy as np
import six.moves.urllib as urllib
import os
import zipfile
import scipy.io as sio



def get_masks_and_images(base_path, dir):
    print(f"> Getting masks from {dir}")
    image_path_array = []
    files  = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if (f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)
                files.append(f)

    #sort image_path_array to ensure its in the low to high order expected in polygon.mat
    zip_list = zip(image_path_array, files)
    sorted_pairs = sorted(zip_list)
    tuples = zip(*sorted_pairs)
    image_path_array, file_name = [list(tuple) for tuple in tuples]

    # Contains segmentation info for each 100 frames of 1 dir
    boxes = sio.loadmat(base_path + dir + "/polygons.mat")
    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    os.makedirs('../data/images', exist_ok=True)
    os.makedirs('../data/masks', exist_ok=True)
    for pointindex, first in enumerate(polygons):
        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)
        # Save images in ../data/images
        cv2.imwrite(f'../data/images/{file_name[pointindex]}', img)
        save_hand = False
        mask = np.zeros((img.shape[0], img.shape[1]))
        for nr_hand, pointlist in enumerate(first):
            has_hand = False
            pts = np.empty((0, 2), int)
            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    save_hand= True
                    has_hand = True
                    x = int(point[0])
                    y = int(point[1])
                    findex += 1
                    append = np.array([[x, y]])
                    pts = np.append(pts, append, axis=0)

            # Fill polynomials around hands
            if has_hand:
                print(nr_hand + 1)
                mask = cv2.fillPoly(mask, [pts], nr_hand+1)
        # Save masks in ../data/masks
        np.save(f'../data/masks/{file_name[pointindex]}', mask)
        print(f"> Saved mask for image {pointindex}")

def generate_derivatives(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_masks_and_images(image_dir, dir)


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
    print("Egohands dataset already downloaded.")
    rename_file=False
    if not os.path.exists("../data/source/egohands"):
        rename_file = True
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("../data/source/egohands")
        print("> Extraction complete")
        zip_ref.close()
    rename_files("../data/source/egohands/_LABELLED_SAMPLES/", rename_file)

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


EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
EGO_HANDS_FILE = "../data/source/egohands_data.zip"


download_egohands_dataset(EGOHANDS_DATASET_URL, EGO_HANDS_FILE)