"""


"""
import cv2
import numpy as np
import six.moves.urllib as urllib
import os
import zipfile
import torch
import scipy.io as sio
import numpy as np


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





def convert_to_segment(base_path, dir):

    print(f"> Getting masks from {base_path + dir}")
    image_path_array = []
    file_name = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if (f.split(".")[1] == "png"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)
                file_name.append(f)

    # Segment images into a different folder using trained model
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model_path = 'model.torch'
    # model = torch.load(model_path)
    # model = torch.nn.DataParallel(model).to(device)
    # model.eval()
    # print(f'Using {torch.cuda.device_count()} GPUs')
    os.makedirs('../data/masks_class', exist_ok=True)
    # Create label csv file
    csv_label = np.zeros(len(image_path_array))
    csv_id = []
    for i, img_path in enumerate(image_path_array):
        # img = np.load(img_path)
        # img = torch.from_numpy(img)
        # with torch.no_grad():
        #     prediction = model([img.to(device)])
        # cv2.imwrite(f'../data/masks_class/{file_name[i]}', prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        print(file_name[i])
        csv_label[i] =  file_name[i][4:5]
        csv_id.append(file_name[i])
    return csv_id, csv_label

def generate_derivatives(image_dir, sets):
    label = []
    id = []
    for set in sets:
        for root, dirs, filenames in os.walk(image_dir):
            for dir in dirs:
                tmp_id, tmp_label = convert_to_segment(image_dir+set+'/', dir)
                label.append(tmp_label)
                id.append(tmp_id)

# rename image files so we dont have overlapping names
def rename_files(image_dir, rename_file):
    sets = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
    if rename_file:
        print("Renaming files")
        loop_index = 0
        for set in sets:
            for root, dirs, filenames in os.walk(image_dir+set):
                for dir in dirs:
                    for f in os.listdir(image_dir+ set+  '/' + dir):
                        if (dir not in f):
                            if(f.split(".")[1] == "png"):
                                loop_index += 1
                                os.rename(image_dir +set + '/'+ dir + "/"  +
                                          "/" + f, image_dir + set +'/' + dir +
                                          "/" + set + "_" + dir + "_" + f)
                        else:
                            break
    generate_derivatives("../data/source/egohands/_LABELLED_SAMPLES/", sets)

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
            "> downloading Leap dataset.")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, dataset_path)
        print("> download complete")
        extract_folder(dataset_path)
    else:
        extract_folder(dataset_path)


LEAP_DATASET_URL = "https://drive.upm.es/index.php/s/efpozEx0D9tQhQa/download"
LEAP_FILE = "../data/source/leap_database.rar"


download_egohands_dataset(LEAP_DATASET_URL, LEAP_FILE)
