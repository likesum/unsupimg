#!/bin/bash

# Download blur kernels
wget https://github.com/likesum/unsupimg/releases/download/v1.0/blur_kernels.zip
unzip blur_kernels.zip -d data
rm blur_kernels.zip
