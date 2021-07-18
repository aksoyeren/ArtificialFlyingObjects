#!/bin/bash

unzip_from_link() {
    local download_link=$1
    local directory=$2
    mkdir -p $directory
    cd $directory
    echo "Downloading to.. " $directory
    aria2c --file-allocation=none --summary-interval=0 "${download_link:-}" \
    && unzip -q -o -d "$directory" \*.zip \
    && rm -rf \*.zip \   
}

eval "$(conda shell.bash hook)"
conda activate DL-labs

echo "Download train set.."

unzip_from_link "http://images.cocodataset.org/zips/train2017.zip" $1

echo "Download validation set.."
unzip_from_link "http://images.cocodataset.org/zips/val2017.zip" $1


echo "Download train/validation annotations.."
unzip_from_link "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" $1

echo "Download finished!"