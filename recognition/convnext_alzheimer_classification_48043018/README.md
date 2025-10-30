# Classification of Alzheimer's Disease using ConvNeXt

**Author**: Cyrus Forudi (48043018)

---

## Project Overview

The aim of this project is to build a classifier for Alzheimer's Disease using the MRI brain scans in the ADNI dataset. By using the latest vision models, namely ConvNeXt, we aim to classify between between two classes:

- Normal Control (NC); and
- Alzheimer's Disease (AD).

with a goal of at least 80% accuracy on the test data.

The selected model is ConvNeXt-T (Tiny), a small sized model appropriate for the amount of images in the ADNI dataset. It is **not** pre-trained on the ImageNet as in the paper, but trained from scratch on the ADNI data.

The ConvNeXt model emerged recently to combat the idea that "Transformers are better at Computer Vision than Convolutional Neural Networks". Liu et al wanted to build a pure convolutional network that could perform at the same standard as a transformer model.

This was achieved by mimicking the concept of self-attention but using only convolution. The ConvNeXt architecture began with a ResNet-50 model and modernised this through 5 design decisions as seen below:

1. macro design
2. ResNeXt
3. inverted bottleneck
4. large kernel size
5. various layer-wise micro designs

![5 design decisions that convert ResNet-50 to ConvNext](images/resnet-50-to-convnext.png)

---

## Table of Contents

- [Project Overview](#project-overview)
- [References](#references)

---

## Reproducibility and Dependencies

All dependencies are found in `environment.yml` file. The key dependencies are:

- torch==2.8.0
- torchaudio==2.8.0
- torchvision==0.23.0

To create the environment run:

```bash
    conda env create -f environment.yml
    conda activate torch
```

## Running the Project

Assuming you have the ADNI dataset.

1. Modify the `MACHINE` variable in `train.py` and `predict.py` to reflect whether you are running on your local machine, Google Colab, or Rangpur. This is to adjust the file paths and the number of allowed worker threads depending on the platform. (Reported training was conducted on Google Colab)

2. Execute the following commands:

    ```bash
    python3 train.py
    python3 predict.py
    ```

When `train.py` is run, it will output a file called `model.pth` which contains the trained model weights. This will then be used by `predict.py` to run inference on the testing data.

## Model Architecture

### Model Selection

When comparing the different sizes in the ConvNeXt model family,
5 different variants of the ConvNeXt model were considered:

- ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
- ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
- ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
- ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
- ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

The ConvNeXt-T (tiny) architecture was chosen as it is most suitable for the size of the limited dataset. Smaller datasets are common in medical imaging due to limitations such as cost and privacy.

![graph of training loss and validation loss](images/loss_graph.png)

![graph of validation accuracy](images/accuracy_graph.png)

## Data

There are 2 classes in the provided ADNI dataset:

- Normal Control (NC); and
- Alzheimer's Disease (AD).

The original data provided was only split into 2 sets, training and testing. However, in machine learning it is beneficial to also have a validation set for tuning hyperparameters. To solve this, the full training set was separated with an 80/20 split into training and validation images respectively. The training data was split rather than the testing data in order to keep the test set completely isolated and avoid the risk of data leakage and overfitting.

This resulted in the following dataset sizes:

*30,590* total images

- Training Set: (total images 17,262)
  - NC : 8958 images
  - AD : 8304 images

- Validation Set: (total images 4,315)
  - NC : 2191 images
  - AD : 2124 images

- Testing Set: (total images 9,013)
  - NC : 4540 images
  - AD : 4473 images

The resulting split is therefore approximately 57% training, 14% validation, and 29% testing.
This ensures that the majority of the data is dedicated to the training of the model, while there is some used for training hyperparameters, and a reasonable amount left at the end for unbiased testing of the performance of the model.

## Training

### Training Data Preprocessing

explain this

``` python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),       # small rotations
    transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
    transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # mild intensity jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalise the image
])
```

### Testing Data Preprocessing

explain this

``` python
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalise the image
])
```

## Results

## Evaluation

## Improvements and Extensions

## Conclusion

## References

1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. ArXiv:2201.03545 [Cs]. [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

2. facebookresearch. (2025). ConvNeXt/models/convnext.py at main · facebookresearch/ConvNeXt. GitHub. [https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py)

3. Muhammad Ardi. (2025, May 6). ConvNeXt Paper Walkthrough: The CNN That Challenges ViT | Towards Data Science. Towards Data Science. [https://towardsdatascience.com/the-cnn-that-challenges-vit/](https://towardsdatascience.com/the-cnn-that-challenges-vit/https://towardsdatascience.com/the-cnn-that-challenges-vit/)

Github Copilot and GPT-5 were used to speed up the development of this project.
