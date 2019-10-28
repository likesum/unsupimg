#!/bin/bash

# Download BSD68
svn checkout https://github.com/cszn/IRCNN/trunk/testsets/BSD68/ data/BSD68
find "$(pwd)/data/BSD68" -name "*.png" > data/BSD68.txt

# Download Set11
svn checkout https://github.com/jianzhangcs/ISTA-Net/trunk/Test_Image data/Set11
find "$(pwd)/data/Set11" -name "*.tif" > data/Set11.txt

