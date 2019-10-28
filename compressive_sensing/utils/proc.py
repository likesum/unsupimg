import numpy as np
import scipy.io
import tensorflow as tf

def gen_mat(insz, outsz, seed):
    mat = np.random.RandomState(seed).randn(outsz, insz)
    mat = mat / np.sqrt(outsz)
    return np.float32(mat)


def load_mat(ratio, path=None):
    if path is None:
        path = 'data/mats/phi_0_%02d_1089.mat'%int(100*ratio)
    mat = scipy.io.loadmat(path)['phi']
    return np.float32(mat)


def get_noisy_mat(mat, nstd, seeds, nblock=100):
    noise = []
    for i in range(seeds.get_shape().as_list()[0]):
        n = tf.contrib.stateless.stateless_random_normal((nblock,)+mat.shape, seeds[i], dtype=tf.float32) * nstd / np.sqrt(mat.shape[-1])
        noise.append(n)
    noise = tf.concat(noise, axis=0)
    mat = mat[None] + noise
    return mat


def compress(imgs, mat, proxy=True):
    b, csz, _, c = imgs.get_shape().as_list()
    assert c == 1

    imgs = tf.reshape(imgs, [-1,1,csz*csz])
    if len(mat.shape) == 2:
        mat = tf.expand_dims(mat, axis=0)
    signal = tf.reduce_sum(imgs*mat, axis=2)  # [b,outsz]

    if not proxy:
        return signal
    else:
        proxy = tf.reduce_sum(signal[:,:,tf.newaxis]*mat, axis=1)
        proxy = tf.reshape(proxy, [-1,csz,csz,1])
        return signal, proxy


def get_proxy(signal, mat):
    csz = int(np.sqrt(mat.shape[-1]))
    if len(mat.shape) == 2:
        mat = tf.expand_dims(mat, axis=0)
    proxy = tf.reduce_sum(signal[:,:,tf.newaxis]*mat, axis=1)
    proxy = tf.reshape(proxy, [-1,csz,csz,1])
    return proxy


def compress_noisy(imgs, mat, nstd, seeds=None, bsz=12, proxy=True):
    b, csz, _, c = imgs.get_shape().as_list()
    outsz = mat.shape[0]
    assert c == 1

    imgs = tf.reshape(imgs, [-1,1,csz*csz])
    mat = tf.expand_dims(mat, axis=0)
    signal = tf.reduce_sum(imgs*mat, axis=2)  # [b,outsz]

    if seeds is None:
        signal = signal + tf.random_normal(tf.shape(signal), stddev=nstd)
    else:
        noise = []
        nblock = b // bsz
        for i in range(bsz):
            n = tf.contrib.stateless.stateless_random_normal([nblock, outsz], seeds[i], dtype=tf.float32) * nstd #  noise for blocks from the same image will be from the seed of the image
            noise.append(n)
        signal = signal + tf.concat(noise, axis=0)

    if not proxy:
        return signal
    else:
        proxy = tf.reduce_sum(signal[:,:,tf.newaxis]*mat, axis=1)
        proxy = tf.reshape(proxy, [-1,csz,csz,1])
        return signal, proxy


def extract_blocks(imgs, csz=33):
    blocks = tf.space_to_depth(imgs, csz)
    return tf.reshape(blocks, [-1,csz,csz,1])


def group_blocks(blocks, imsz=330, csz=33):
    blocks = tf.reshape(blocks, [-1, imsz//csz, imsz//csz, csz*csz])
    return tf.depth_to_space(blocks, csz)


def shape(t):
    print(t.get_shape().as_list())
    exit()