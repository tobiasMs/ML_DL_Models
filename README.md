# TOMATO DEEP LEARNING MODELS

This project contains deep learning experiments for tomato leaf classification, mainly to distinguish between healthy and diseased leaves. The workflow starts with the tomato dataset, applies data augmentation to create a binary dataset, and then trains several transfer learning architectures such as EfficientNet, MobileNetV3, and ResNet.

## Dataset

The source dataset can be downloaded from Kaggle here:

https://www.kaggle.com/datasets/ashishmotwani/tomato

The dataset is used as a multiclass dataset in the `TomatoDataset/` folder. After that, `data_augmentation.py` builds a binary dataset in `TomatoBinaryDataset/` with two classes:

- `healthy`
- `diseased`

## Project Structure

```text
ML_DL_Models/
├── data_augmentation.py
├── README.md
├── DL/
│   ├── requirements.txt
│   ├── EfficientNet/
│   │   ├── Base/
│   │   └── Opt/
│   ├── MobileNetV3/
│   │   ├── Base/
│   │   ├── Basev2/
│   │   └── Opt/
│   └── ResNet/
│       ├── Base/
│       ├── Basev2/
│       └── Opt/
├── TomatoDataset/
└── TomatoBinaryDataset/
```

Main folder overview:

- `TomatoDataset/` contains the original Kaggle dataset in multiclass format.
- `TomatoBinaryDataset/` contains the processed and augmented binary classification dataset.
- `DL/` contains model training scripts and saved model files.
- `data_augmentation.py` is used to convert the multiclass dataset into a binary dataset.

## Python Modules Used

### External libraries

The libraries used in this project and listed in `DL/requirements.txt` are:

- `tensorflow` for building, training, and saving deep learning models.
- `numpy` for numerical operations and array manipulation.
- `matplotlib` for training curve and result visualization.
- `seaborn` for confusion matrix visualization.
- `scikit-learn` for `train_test_split`, `classification_report`, `confusion_matrix`, and `class_weight`.
- `opencv-python` for reading and saving images during augmentation.
- `albumentations` for the image augmentation pipeline.
- `tqdm` for progress bars during copy and augmentation steps.

### Python standard library

The built-in Python modules used in this project are:

- `pathlib` for file and folder path management.
- `random` for random image selection during augmentation.
- `shutil` for removing the old output folder before generating a new dataset.
- `datetime` for recording training time.
- `platform` for storing system information during training.

## Main Scripts

### `data_augmentation.py`

This script reads all images from `TomatoDataset/`, separates the classes into `healthy` and `diseased`, and then:

1. copies the original images into the output folder,
2. applies augmentation using `albumentations`,
3. balances the number of images until the target is reached,
4. saves the final result to `TomatoBinaryDataset/`.

### Training scripts in `DL/`

All training scripts follow a similar workflow:

- read the dataset from `TomatoBinaryDataset/`,
- perform a stratified train-validation split,
- build a transfer learning model,
- train the model and save the best result,
- generate a training report, confusion matrix, and training curve.

Available models:

- EfficientNetB0
- MobileNetV3Large
- ResNet50

## Installation

It is recommended to create a virtual environment first, then install the dependencies:

```bash
pip install -r DL/requirements.txt
```

## How to Run

### 1. Prepare the dataset

Download the Kaggle dataset and extract it into the `TomatoDataset/` folder.

### 2. Build the binary dataset

Run:

```bash
python data_augmentation.py
```

The generated dataset will be saved to `TomatoBinaryDataset/`.

### 3. Train a model

Run the model script you want from the `DL/` folder, for example:

```bash
python DL/EfficientNet/Base/efficientnet_base.py
python DL/MobileNetV3/Base/mobilenet_base.py
python DL/ResNet/Base/resnet_base.py
```

## Training Outputs

Each training script usually produces the following files in its model folder:

- best model file with `.keras` extension
- final model file with `.keras` extension
- training report in `.txt` format
- confusion matrix in `.png` format
- training curve in `.png` format

## Notes

- Make sure the dataset folder structure matches what the scripts expect.
- The binary classification training scripts expect `diseased/` and `healthy/` folders inside `TomatoBinaryDataset/`.
- Some output filenames differ between model folders because they were created for different experiments.
