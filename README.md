
# Hierarchical Deep Temporal Models for Group Activity Recognition

![Group activity recognition via a hierarchical model](assets/hierarchical-model-visualize.jpeg "Group activity recognition via a hierarchical model")

This repo is an implementation for this paper (Mostafa et al., [2016](https://arxiv.org/abs/1511.06040))

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

### Setup your Kaggle API credentials

1) Log into your Kaggle account (or create one).

2) Go to your [settings](https://www.kaggle.com/settings) and generate a new API token.

3) A `kaggle.json` file will be downloaded.

4) Copy this file to `your-project-folder/.config/kaggle`.

5) For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:
```bash
chmod 600 ~/.config/kaggle/kaggle.json
```

### Downloading the dataset

```bash
$ chmod +x scripts/download-dataset.sh
$ scripts/download-dataset.sh
```
