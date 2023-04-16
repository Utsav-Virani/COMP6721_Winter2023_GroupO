# Flower Classification

## Overview
This project involves the classification of images of flowers using deep learning models. The team experimented with several popular models such as ResNet, VGGNet, and MobileNet to improve the accuracy of image classification tasks. The project was developed as part of COMP6721 in the Winter 2023 semester.
<br />
<br />
<hr />

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
│   ├── mobilenet-7.ipynb
│   ├── mobilenet-16.ipynb
│   ├── SGD_Optimizer_vggnet-7.ipynb
│   ├── transfer_resnet_5.ipynb
│   ├── transfer_mobilenet_16.ipynb
│   └── transfer_vggnet-5.ipynb
├── LICENSE
├── .gitignore
├── README.md
└── requirements.txt
```
### Notebooks
This folder contains `Jupyter notebooks` used for machine learning experiments with various models. Each notebook is named after the corresponding deep learning model used for the experiments. There are multiple versions of each notebook, with different numbers indicating the depth of the neural network used for the experiments. The notebooks include code for loading and preprocessing datasets, defining and training neural network models, and evaluating the performance of the models. They also contain visualizations of training progress and model performance metrics.

These notebooks are compatible with `Google Colab`, `Jupyter Notebooks`, and `Kaggle`. To use these notebooks, you `must install all the required libraries` mentioned in the ```requirements.txt``` file in your environment and place the datasets in the appropriate location, or adjust the folder path mentioned in the code accordingly.

Docs
This folder contains all the project documents created by the team. These include the project proposal and the final project report.


### LICENSE:
This file contains the license under which the software in this repository is released. It is important to read and understand the license before using or distributing any code or other materials in this repository.

### requirements.txt:
This file contains a list of all the Python packages that are required to run the notebooks in this repository. To use these notebooks, you should install these packages in a virtual environment or in your system's Python installation. <br >You can do this by running the following command in your terminal or command prompt:

``` 
pip install -r requirements.txt
```

### .gitignore:
This file tells Git which files and directories to ignore when committing changes to the repository. Files and directories listed in this file will be excluded from version control and will not be pushed to the remote repository. This can be useful for excluding large files, temporary files, log files, or other files that do not need to be included in the repository.

<hr />

## Summary of the generated models:

|               | Resnet-18 |              |        | Mobilenet-V2 |              |        | VGG-19    |              |        |
| ------------- | --------- | ------------ | ------ | ------------ | ------------ | ------ | --------- | ------------ | ------ |
|               | D1        | D2           | D3     | D1           | D2           | D3     | D1        | D2           | D3     |
| Accuracy (%)  | 82.99     | 87.14        | 81.98  | 82.99 | 87.59 | 81.85 | 10.13 | 14.29 | 6.68 |
| Precision (%) | 83        | 88           | 83     | 83 | 88 | 82 | 1 | 2 | 0 |
| Recall (%)    | 83        | 87           | 82     | 83 | 88 | 82 | 10 | 14 | 7 |
| F1 score      | 83        | 87           | 82     | 83 | 88 | 82 | 20 | 4 | 1 |

