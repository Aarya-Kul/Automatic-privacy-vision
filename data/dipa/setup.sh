#!/usr/bin/env bash

echo "downloading DIPA files from https://anranxu.github.io/DIPA_visualization/"

wget --no-check-certificate dipa-download.s3.ap-northeast-1.amazonaws.com/dataset.zip
unzip dataset.zip
cp -r dataset/{images,annotations,original\ labels} .

rm dataset.zip
rm -rf dataset

echo "images, annotations, and labels donwloaded from DIPA dataset"
