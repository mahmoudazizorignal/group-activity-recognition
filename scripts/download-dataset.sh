#!/bin/bash

DATASET_PATH="data/volleyball"

if test -d "$DATASET_PATH"; then

    echo -e "\nDataset is already downloaded. Do you want to redownload it? (y/n)"
    read answer

    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then

        echo -e "\nDownloading skipped."
    
    else

        echo -e "\nRe-Downloading Dataset ..."
        kaggle datasets download ahmedmohamed365/volleyball -p "$DATASET_PATH" --unzip
        echo -e "\nDataset downloaded successfully!"

    fi

else

    echo -e "\nDownloading Dataset ..."
    kaggle datasets download ahmedmohamed365/volleyball -p "$DATASET_PATH" --unzip
    echo -e "\nDataset downloaded successfully!"

fi