import os
from pathlib import Path
import zipfile
import torch

if not (Path("../dataset/flower-classification-5-classes-roselilyetc.zip").is_file() or Path("../dataset/flowerClass.zip").is_file()):
    os.system("kaggle datasets download -d utkarshsaxenadn/flower-classification-5-classes-roselilyetc -p ../dataset/")
    os.system("ren ..\\dataset\\flower-classification-5-classes-roselilyetc.zip flowerClass.zip")

path_to_flowerClass = "../dataset/flowerClass.zip";
if not Path("../dataset/flowerClass").is_dir():
    with zipfile.ZipFile(path_to_flowerClass, 'r') as zip_ref:
        zip_ref.extractall("../dataset/flowerClass")


print("Is CUDA available in the device.: {}".format(torch.cuda.is_available()))

print("ALL SET!!!")
