# Machine Learning Museum Dataset Classification - Phase 1

## Introduction
The Machine Learning Museum Dataset Classification is a project that aims to classify images of museum artifacts into their respective categories as indoor or outdoor. The dataset consists of images of museum artifacts. The project is divided into two phases. The first phase involves the classification of images into indoor and outdoor categories using the following standard 
machine learning algorithms:
- Decision Tree
- Random Forest
- Boosting (Gradient Boosting)

## Dataset
The dataset consists of images of museum artifacts. The dataset is divided into two categories: indoor and outdoor. The dataset is divided into two folders: indoor and outdoor. The indoor folder contains images of museum artifacts that are indoors, while the outdoor folder contains images of museum artifacts that are outdoors.
The dataset can be downloaded from the following link [Museum Dataset](https://drive.google.com/drive/folders/1bDGTc0drcpUKslyM6nB9u1XSl1HF7RmP?usp=drive_link). Place the dataset in the `data` folder.
with the following structure:
```
data
- train
  - indoor
    - image1.jpg
    - image2.jpg
    - ...
  - outdoor
    - image1.jpg
    - image2.jpg
    - ...
```

## Requirements
- Python 3.12.2

## Installation
1. Clone the repository
```bash
git clone git@github.com:dbaeka/comp6721-p1.git
```
2. Install the required libraries
```bash
pip install -r requirements.txt
```

## Usage
1. Navigate to the project directory
```bash
cd comp6721-p1
```
2. Run the following command to train the model
```bash
python train.py
```
3. Run the following command to test the model
```bash
python test.py
```

## Results
The results of the classification are displayed in the form of a confusion matrix and classification report. The classification report displays the precision, recall, f1-score, and support for each class. The confusion matrix displays the true positive, false positive, true negative, and false negative values for each class.

## Conclusion
The project aims to classify images of museum artifacts into their respective categories as indoor or outdoor. The project uses standard machine learning algorithms to classify the images. The results of the classification are displayed in the form of a confusion matrix and classification report. The project can be further extended to include more categories and more advanced machine learning algorithms.