# Training Image Estimators without Image Ground-Truth --- Face Deblurring

This directory contains our training and testing code for face deblurring, including our supervised baseline, nonblind unsupervised method and blind unsupervised method (both clean images and blur kernels are unavailable). Note that while we use clean face images to synthesize the blurry input, for our unsupervised method, the loss is computed without the clean face images (and without GT kernels for the blind training).

Please run all the scripts from this working directory. See the code or run
```
./python_file.py -h
```
for a list of parameters you can specify for each python script. 

## Testing
- Please download the Helen and CelebA deblurring test set from the paper [DSFD](https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur) and unzip them to the folder `data/`.
- Our pre-trained models, including supervised, unsupervised nonblind and blind models, are provided [here](https://github.com/likesum/unsupimg/releases/download/v1.0/deblur_models.zip). You can download and unzip them by running
```
./scripts/download_models.sh
```
- Test these pre-trained models by
```
./test.py [--data Helen/CelebA_folder] [--path path_to_model] [--outpath save_outputs_to]
```

## Training
### Prepare the training data
- We use face images from the training split of Helen dataset and CelebA dataset to train our models. You can download the dataset from their official webpages ([Helen](http://www.ifp.illinois.edu/~vuongle2/helen/) and [Aligned&Cropped CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).
- Since the [DSFD benchmark](https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur) consists of 128 x 128 images with the face approximately centered, we also preprocess our training set. Specifically, for CelebA dataset, we use the Aligned&Cropped version of the dataset and further center crop the images to 128 x 128. For Helen dataset, the images are not aligned so we use segmentation maps provided in the dataset as references to crop the image so that the face is approximately in the center. We then resize images to 128x128. Please refer to `scripts/crop_celebA.py` and `scripts/crop_Helen.py` for example scripts on how we preprocess the training data.
- Create the files `data/celebA_train.txt`, `data/celebA_dev.txt`, `data/Helen_train.txt` and `data/Helen_dev.txt` with lists of image paths to these preprocessed images, which will be used for training and validation.
- Download the blur kernels we used for training by
```
./scripts/download_kernels.sh
```

### Training our supervised baseline
To train our supervised baseline by using ground-truth images, run 
```
./train_supervised.py
```
Trained models (including checkpoints) and training logs will be saved in `wts/supervised`.

### Non-blind unsupervised training
To train the model without using ground-truth images, but assuming the blur kernels for the training blurry images are known, run
```
./train_nonblind.py
```
Trained models (including checkpoints) and training logs will be saved in `wts/nonblind`.

### Blind unsupervised training
To train the model without using ground-truth images or the blur kernels for the training blurry images, run
```
./train_blind.py
```
Trained models (including checkpoints) and training logs will be saved in `wts/blind`.

### Note
- You can press `ctrl-c` at any time to stop the training and save the checkpoints (model weights and optimizer states). The training script will resume from the latest checkpoint (if any) in the model directory and continue training.
- For the supervised baseline, we blur each training image with a random blur kernel on the fly and randomly add noise to synthesize the blurry measurements. But for our unsupervised method, we randomly select two fixed blur kernels for each image resulting in a pair of blurry measurements. We also fix the random noise for these two images.
