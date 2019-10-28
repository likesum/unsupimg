import os
from glob import glob
from imageio import imread, imsave

nms = glob('data/celebA/img_align_celeba_png/*.png')
outdir = 'data/celebA/cropped/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

i = 0
for nm in nms:
    img = imread(nm)
    outnm = outdir + os.path.basename(nm)
    imsave(outnm, img[68:68+128,25:25+128,:])

    i = i + 1
    if i % 10000 == 0:
        print("Finished %d images"%i)