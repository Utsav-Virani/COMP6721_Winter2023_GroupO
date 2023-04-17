#  Flower Classification

## Overview
This project involves the classification of images of flowers using deep learning models. The team experimented with several popular models such as ResNet, VGGNet, and MobileNet to improve the accuracy of image classification tasks. The project was developed as part of COMP6721 in the Winter 2023 semester.
<br />
<br />
<hr />
Video presentation of the project : <a href="https://drive.google.com/file/d/1WeUqoijLtB-4HBbe2iwRjDs2WH0kxQH0/view?usp=sharing"> Google Drive </a> <br />
Presentation Slides: <a href="https://drive.google.com/file/d/1SoPH9GDakJo5O5cO07lee23vc1H11Zl4/view?usp=share_link"> Google Drive </a>
<br />
<hr />

## Datasets
Datasets | Auther | Links |
| ------------- | --------- | ------------ |
| Flower Classification | utkarshsaxenadn | <a href="https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc">*kaggle*</a>, <a href="https://drive.google.com/file/d/129efP8KotN_J4xleFU6rnfYfeMnwLEm2/view?usp=share_link">*Google Drive*</a> |
| 🌸 \| Flowers | l3llff | <a href="https://www.kaggle.com/datasets/l3llff/flowers">*kaggle*</a>, <a href="https://drive.google.com/file/d/18624X71QkDaRGt4_3kD653odU4HU1BbL/view?usp=share_link">*Google Drive*</a> |
| flowers | nadyana | <a href="https://www.kaggle.com/datasets/nadyana/flowers">*kaggle*</a>, <a href="https://drive.google.com/file/d/1qWMuBOltrt6HkIjtUnJdY7p0uZ6J05ik/view?usp=sharing">*Google Drive*</a> |

<hr />
<br >

## Pretrained models

- All the Pretrained models are available 
<a href="https://drive.google.com/drive/folders/1HNUYiqfY7ADcNtDRBwwVOoht0UPNmgQw?usp=sharing">*here*</a>

<hr />
<br >

## Folder Structure
The project folder has the following structure:

```
COMP6721_Winter2023_GroupO
├── Docs
│   ├── Project_Proposal_Group_O.pdf
│   └── Project_Report_Group_O.pdf
├── notebooks
│   ├── resnet_5.ipynb
│   ├── resnet_7.ipynb
│   ├── resnet_16.ipynb
│   ├── vggnet-5.ipynb
│   ├── vggnet-7.ipynb
│   ├── vggnet-16.ipynb
│   ├── mobilenet-5.ipynb
│   ├── mobilenet_7.ipynb
│   ├── mobilenet_16.ipynb
│   ├── SGD_Optimizer_vggnet-7.ipynb
│   ├── transfer_resnet_5.ipynb
│   ├── transfer_mobilenet_16.ipynb
│   └── transfer_vggnet-5.ipynb
├── LICENSE
├── .gitignore
├── README.md
└── requirements.txt
```
### <li>Notebooks
This folder contains `Jupyter notebooks` used for machine learning experiments with various models. These notebooks are compatible with `Google Colab`, `Jupyter Notebooks`, and `Kaggle`. 
>**Note:**
>*To use these notebooks, you `must install all the required libraries` mentioned in the ```requirements.txt``` file in your environment and place the datasets in the appropriate location, or adjust the folder path mentioned in the code accordingly.*

### <li>Docs
This folder contains all the project documents created by the team. These include the project proposal and the final project report.


### <li>LICENSE:
This file contains the license under which the software in this repository is released. It is important to read and understand the license before using or distributing any code or other materials in this repository.

### <li>requirements.txt:
This file contains a list of all the Python packages that are required to run the notebooks in this repository. 
>**Note:**
>*You can install the required libraries by running the following command in your terminal or command prompt:*
``` 
pip install -r requirements.txt
```

<hr />

<br />

## How to run the code
>**Note:**
> *If you want to train the model from the scratch follow the bellow steps:*

1. Clone the repository to your local machine using the following command:
```
git clone https://github.com/Utsav-Virani/COMP6721_Winter2023_GroupO
```
2. Install the required dependencies using the following command:
```
pip install -r requirements.txt
```
3. Launch Jupyter Notebook using the following command:
```
jupyter notebook
```
4. Navigate to the notebooks directory in your Jupyter Notebook dashboard and open the `.ipynb` file.

5. Run the notebook by clicking on the "Run" button or by using the "Shift + Enter" keyboard shortcut.

>**Note:**
>*The dataset used in the notebook can be found on the `Kaggle` or `GDrive`. The links for that are mentioned above.*

>**Note:**
>*Its is convineint for you to run this noptebooks on `kaggle` as you dont have to download the datasets or in the `google Colab`.*

<hr />
<br>

## How to Test the model
>**Note:**
> *If you do not want to train the model from the scratch and just have to test the pretrained model then follow the bellow steps:*

1. Load the `'ModelTesting.ipynb'` file from the `'notebooks'` folder, which contains the testing code for the model.

2. Provide the input path of the trained model in the `'torch.load()'` function, which will load the model for testing purposes.

3. A dictionary named `'sat_map'` has been defined for the `dataset-1`, which can be used to convert the predicted output to class labels. Similar dictionaries can be created for other datasets as well.

4. To pass an image for classification, simply provide the path of the image in the `'image.open()'` method, which will open the image for processing.

5. Execute the code to obtain the prediction for the input image.

<br>
<hr />

## Summary of the generated models

|               | Resnet-18 |              |        | Mobilenet-V2 |              |        | VGG-19    |              |        |
| ------------- | --------- | ------------ | ------ | ------------ | ------------ | ------ | --------- | ------------ | ------ |
|               | D1        | D2           | D3     | D1           | D2           | D3     | D1        | D2           | D3     |
| Accuracy (%)  | 82.99     | 87.14        | 81.98  | 82.99 | 87.59 | 81.85 | 10.13 | 14.29 | 6.68 |
| Precision (%) | 83        | 88           | 83     | 83 | 88 | 82 | 1 | 2 | 0 |
| Recall (%)    | 83        | 87           | 82     | 83 | 88 | 82 | 10 | 14 | 7 |
| F1 score      | 83        | 87           | 82     | 83 | 88 | 82 | 20 | 4 | 1 |

