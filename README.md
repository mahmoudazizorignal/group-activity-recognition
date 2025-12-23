
# Hierarchical Deep Temporal Models for Group Activity Recognition

<center>
    <img src="assets/hierarchical-model-visualize.jpeg">
</center>


This repo is a modern implementation of CVPR16 ["Hierarchical Deep Temporal Models for Group Activity Recognition"](https://arxiv.org/pdf/1511.06040) paper using PyTorch.

## Requirements

- Python 3.12 or later

## Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
2) Create a new environment using the following commmand:
```bash
$ conda create -n activity-recognition python=3.12
```
3) Activate the environment:
```bash
$ conda activate activity-recognition
```


## (Optional) Setup your command line for better readability
```bash
$ export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```


## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```
Set your environment variables in the `.env` file.

### Downloading the dataset

```bash
$ kaggle datasets download ahmedmohamed365/volleyball -p ./data --unzip
```

## Baselines
We implemented all the baselines in the paper but with only one difference, the paper used AlexNet as the feature extractor, and we used ResNet-50. So, our results is expected to be better than those in the paper.

- B1 **Image Classification**: This baseline is the basic ResNet-50 model fine-tuned for group activity recognition in a single frame.

- B3 **Fine-tuned Person Classification**: Here, we use the ResNet-50 that fine-tuned to recognize person-level actions. Then, the final fully connected layer is pooled over all players to recognize group activities in a scene without any fine-tuning of the ResNet-50 model (we freeze it). The rationale behind this baseline is to examine a scenario where person-level action annotations as well as group activity annotations are used in a deep learning model that does not model the temporal aspect of group activities. This is very similar to the two-stage model without the temporal modeling.

- B4 **Temporal Model with Image Features**: This baseline is a temporal extension of the B1. It examines the idea of feeding image level features directly to an LSTM model to recognize group activities. In this baseline, the ResNet-50 model is deployed on the whole image and
resulting final fully connected layer's features are fed to an LSTM model.

- B5 **Temporal Model with Person Features**: We use ResNet-50 on the person-level actions, and the resulting final fully connected layer's are pooled over all players, and then feed to an LSTM model to recognize group-activities

- B6 **Two-stage Model without LSTM 1**: This baseline is a variant of the final model, omitting the person-level temporal model (LSTM 1). Instead, the person-level classification is done only with the fine-tuned person CNN (ResNet-50).

- B7 **Two-stage Model All Pooling**: We use ResNet-50 for feature extraction of the persons crops, we then pass them to the first LSTM model to do classification for the players-level actions. Then, for each frame, we max pool all the players in the scene, and we pass them to the second LSTM model to do classification for the group-level actions.

- B8 **The Final Two-stage Model**: Same as the pervious baselines except for one trick. Instead of max pooling all the players in the scene, we first sort the bounding boxes of each player by position of the top left point to ensure that all the left team members are at the begining and the right team members are at the end. We then max pool the two groups independently from each other, and we concatenate them and pass them to the second LSTM model for group-level actions.

## Results

| baseline | loss    | acc      | f1       |
|----------|---------|----------|----------|
| b1       | 1.13796 | 0.7854   | 0.7877   |
| b3       | 0.7185  | 0.7734   | 0.7729   |
| b4       | 0.7837  | 0.7812   | 0.7755   |
| b5       | 0.5079  | 0.81138  | 0.80093  |
| b6       | 0.46214 | 0.836078 | 0.83073  |
| b7       | 0.35916 | 0.88997  | 0.88827  |
| b8       | 0.27161 | 0.925898 | 0.923733 |
<br><br>

<div style="text-align: center;">
    <figure style="display: inline-block; margin: 0;">
        <img src="assets/confusion_matrix_b7.png" alt="Confusion Matrix">
        <figcaption> Confusion matrix obtained using the two-stage hierarchical model (B7), using 1 group style for all players</figcaption>
    </figure>
</div>
<br><br>
<div style="text-align: center;">
    <figure style="display: inline-block; margin: 0;">
        <img src="assets/confusion_matrix_b8.png" alt="Confusion Matrix">
        <figcaption> Confusion matrix obtained using the final two-stage hierarchical model (B8), using 2 groups style.</figcaption>
    </figure>
</div>

<br>

You can see that the model was confused between *l_winpoint* and *r_winpoint* when we pooled all the players together. But the confusion dissappeared when we used 2 group style in B8.
