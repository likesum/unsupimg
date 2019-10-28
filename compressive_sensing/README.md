# Training Image Estimators without Image Ground-Truth --- Compressive Sensing

This directory contains our training and testing code for compressive sensing, including our supervised baseline and unsupervised training methods. Note that while we use clean images to synthesize the compressive measurement, for our unsupervised method, the loss is computed using only the measurements.

Please run all the scripts from this working directory. See the code or run
```
./python_file.py -h
```
for a list of parameters you can specify for each python script. 

## Testing
- We evaluated our unsupervised method on two datasets, [Set11](https://github.com/jianzhangcs/ISTA-Net/tree/master/Test_Image) and [BSD68](https://github.com/cszn/IRCNN/tree/master/testsets/BSD68). Download them by running
```
./scripts/download_testset.sh
```
- Our pre-trained models, including supervised and unsupervised models, are provided [here](https://github.com/likesum/unsupimg/releases/download/v1.0/CS_models.zip). You can download and unzip them by running
```
./scripts/download_models.sh
```
- Finally, to test these pre-trained models, run
```
./test.py [--ratio compression_ratio] [--data BSD68/Set11] [--path path_to_model] [--outpath save_outputs_to]
```

## Training
### Prepare the training data
- To train your own model, please modify the files `data/train.txt` and `data/dev.txt` with list of image paths to be used for training and validation. For our paper, we downloaded ImageNet images that has size larger than 363x363 and randomply cropped them to size 363x363. In total, we used 100k images from the ImageNet training set for training, and 256 images from the validation split for validation.
- For the supervised baseline, during training we take random 330x330 crops of the image on the fly. We extract and compress each non-overlapping 33x33 block of the crop and reconstruct the cropped image from them. For our unsupervised method, for each image we take two random but fixed crops of the image and compress each non-overlapping 33x33 block. The compressed vectors of these two slightly shifted images together formulate our two compressive measurements. We compute the "swap-loss" by swapping the overlapping regions of their reconstructions and re-compress (see the code and paper for more details).
- We use the compressing matrices in `data/mats`, which are provided by [ISTA-Net](https://github.com/jianzhangcs/ISTA-Net).

### Training our supervised baseline
To train our supervised baseline by using ground-truth images, run 
```
./train_supervised.py [--ratio compression_ratio]
```
Trained models (including checkpoints) and training logs will be saved in `wts/supervised_ratioN`.

### Training with our unsupervised method
To the model with our unsupervised method by computing the loss using only the measurements, run
```
./train_unsupervised.py [--ratio compression_ratio]
```
Trained models (including checkpoints) and training logs will be saved in `wts/unsupervised_ratioN`.

### Note
You can press `ctrl-c` at any time to stop the training and save the checkpoints (model weights and optimizer states). The training script will resume from the latest checkpoint (if any) in the model directory and continue training.
