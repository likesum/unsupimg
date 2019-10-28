import numpy as np
import tensorflow as tf

def gen_blur(images, kernels, nstd=2, seeds=None):
    '''
    Input a list of 3d tensors for images and kernels
    '''
    nstd = nstd / 255.

    bsz, h, w, c = images.get_shape().as_list()
    kbsz, kh, kw, kc = kernels.get_shape().as_list()
    assert kbsz==bsz and kc==1

    images = tf.transpose(images, [1,2,0,3]) # h x w x bsz x c
    images = tf.reshape(images, [1,h,w,bsz*c])

    kernels = tf.tile(kernels, [1,1,1,c])
    kernels= tf.transpose(kernels, [1,2,0,3]) # kh x kw x bsz x c
    kernels = tf.reshape(kernels, [kh,kw,bsz*c,1])

    padding = [[0,0], [(kh-1)//2,(kh-1)//2], [(kw-1)//2, (kw-1)//2], [0,0]]
    images = tf.pad(images, padding, 'REFLECT')
    blurs = tf.nn.depthwise_conv2d(images, kernels, [1,1,1,1], 'VALID') # 1 x h x w x bsz*c

    blurs = tf.reshape(blurs, [h,w,bsz,c])
    blurs = tf.transpose(blurs, [2,0,1,3])

    if nstd != 0:
        if seeds is None:
            noise = tf.random_normal([bsz,h,w,c],stddev=nstd)
        else:
            noise = []
            for i in range(bsz):
                n = tf.contrib.stateless.stateless_random_normal([h,w,c], seeds[i], dtype=tf.float32) * nstd
                noise.append(n)
            noise = tf.stack(noise)

        blurs = blurs + noise

    return blurs


def est_kernel(blurs, deblurs, nstd=2, ksz=27):
    assert ksz%2 == 1
    hksz = (ksz-1)//2
    if nstd == 0:
        nstd = 1e-6

    blurs = tf.cast(tf.transpose(blurs, [0,3,1,2]), tf.complex64)
    deblurs = tf.cast(tf.transpose(deblurs, [0,3,1,2]), tf.complex64)

    fft_blurs = tf.fft2d(blurs)
    fft_deblurs = tf.fft2d(deblurs)

    numerator = fft_deblurs * tf.conj(fft_blurs)
    denominator = tf.abs(fft_blurs)**2 + (nstd/255.)**2.
    out = tf.real(tf.ifft(numerator/denominator))
    out = tf.transpose(out, [0,2,3,1])

    out1 = tf.concat([out[:,-hksz:,-hksz:], out[:,:hksz+1,-hksz:]], axis=1)
    out2 = tf.concat([out[:,-hksz:,:hksz+1], out[:,:hksz+1,:hksz+1]], axis=1)
    kernels = tf.concat([out1,out2], axis=2)
    kernels = kernels / tf.reduce_mean(kernels, axis=[1,2])

    return kernels

def nest_kernel(blurs, deblurs, nstd=2, ksz=27):
    assert ksz%2 == 1
    hksz = (ksz-1)//2
    if nstd == 0:
        nstd = 1e-6

    # print(blurs.shape)
    # exit()
    blurs = np.transpose(blurs, (0,3,1,2))
    deblurs = np.transpose(deblurs, (0,3,1,2))

    fft_blurs = np.fft.fft2(blurs)
    fft_deblurs = np.fft.fft2(deblurs)

    numerator = fft_deblurs * np.conj(fft_blurs)
    denominator = np.abs(fft_blurs)**2 + (nstd/255.)**2.
    out = np.real(np.fft.ifft2(numerator/denominator))
    out = np.transpose(out, (0,2,3,1))

    out1 = np.concatenate([out[:,-hksz:,-hksz:], out[:,:hksz+1,-hksz:]], axis=1)
    out2 = np.concatenate([out[:,-hksz:,:hksz+1], out[:,:hksz+1,:hksz+1]], axis=1)
    kernels = np.concatenate([out1,out2], axis=2)
    kernels = kernels / np.mean(kernels, axis=(1,2), keepdims=True)

    return kernels








