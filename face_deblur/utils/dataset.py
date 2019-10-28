import os
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, lfile, kfile, bsz, niter=0, isval=False, imsz=[128,128], rand_kernel=False):
        self.lfile = lfile
        self.kfile = kfile
        self.bsz = bsz
        self.imsz = imsz
        self.isrand = not isval
        self.isrand_kernel = rand_kernel

        # Set up fetch graph
        self.graph()

        # Load file list
        if type(lfile) is list:
            self.files = lfile
        else:
            self.files = [l.strip() for l in open(lfile).readlines()]
        self.ndata = len(self.files)

        # Load kernel list
        self.kfiles = [l.strip() for l in open(kfile).readlines()]

        self.niter = niter*bsz

        # Setup shuffling
        if self.isrand and self.isrand_kernel:
            self.rand = np.random.RandomState(0)
            for i in range(niter//self.ndata + 1):
                idx = self.rand.permutation(self.ndata)
                self.kernels = [
                    (self.kfiles[k[0]], self.kfiles[k[1]]) \
                    for k in self.rand.randint(len(self.kfiles), size=(self.ndata,2)) \
                ]
            self.idx = np.int32(idx)
        
        elif self.isrand:
            self.rand = np.random.RandomState(0)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
            # Fixed kernel for each image of unsupervised training set
            self.kernels = [
                (self.kfiles[k[0]], self.kfiles[k[1]]) \
                for k in np.random.RandomState(1234).randint(len(self.kfiles), size=(self.ndata,2)) \
            ]

        else:
            self.idx = np.int32(np.arange(self.ndata))
            # Fixed kernel for each image of val set
            self.kernels = [
                (self.kfiles[k[0]], self.kfiles[k[1]]) \
                for k in np.random.RandomState(1234).randint(len(self.kfiles), size=(self.ndata,2)) \
            ]

        self.random_seeds = np.random.RandomState(1234).randint(0, 65536, size=(self.ndata,4), dtype=np.int32)


    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            fd[self.names[i]] = self.files[idx]
            fd[self.knames1[i]] = self.kernels[idx][0]
            fd[self.knames2[i]] = self.kernels[idx][1]
            fd[self.seeds[i]] = self.random_seeds[idx]

            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
                if self.isrand and self.isrand_kernel:
                    self.kernels = [
                        (self.kfiles[k[0]], self.kfiles[k[1]]) \
                        for k in self.rand.randint(len(self.kfiles), size=(self.ndata,2)) \
                    ]
            self.niter = self.niter + 1

        return fd


    def graph(self):
        self.names, self.knames1, self.knames2, self.seeds = [], [], [], []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))
            self.knames1.append(tf.placeholder(tf.string))
            self.knames2.append(tf.placeholder(tf.string))
            self.seeds.append(tf.placeholder(tf.int32, shape=[4]))

        images, kernels1, kernels2 = [], [], []
        for i in range(self.bsz):
            image = tf.read_file(self.names[i])
            image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)

            kernel1 = tf.read_file(self.knames1[i])
            kernel1 = tf.image.decode_png(kernel1, channels=1, dtype=tf.uint16)
            kshape = tf.shape(kernel1)
            npad = (27-kshape[0:2])/2
            padding = tf.cast(tf.stack([npad,npad,[0,0]], axis=0), tf.int32)
            kernel1 = tf.pad(kernel1, padding, 'constant')

            kernel2 = tf.read_file(self.knames2[i])
            kernel2 = tf.image.decode_png(kernel2, channels=1, dtype=tf.uint16)
            kshape = tf.shape(kernel2)
            npad = (27-kshape[0:2])/2
            padding = tf.cast(tf.stack([npad,npad,[0,0]], axis=0), tf.int32)
            kernel2 = tf.pad(kernel2, padding, 'constant')

            images.append(image)
            kernels1.append(kernel1)
            kernels2.append(kernel2)

        images = tf.to_float(tf.stack(images)) / 255.
        images.set_shape([self.bsz,self.imsz[0],self.imsz[1],3])

        kernels1 = tf.to_float(tf.stack(kernels1)) / (2**16-1.0)
        kernels2 = tf.to_float(tf.stack(kernels2)) / (2**16-1.0)
        kernels1.set_shape([self.bsz,27,27,1])
        kernels2.set_shape([self.bsz,27,27,1])

        outs = [images, kernels1, kernels2, tf.stack(self.seeds)]

        # Fetch op
        self.batch, self.fetchOp = [], []
        for i in range(len(outs)):
            var = tf.Variable(tf.zeros(outs[i].shape,dtype=outs[i].dtype),trainable=False)
            self.batch.append(var)
            self.fetchOp.append(tf.assign(var, outs[i]).op)
        self.fetchOp = tf.group(self.fetchOp)


# Sets up a common batch variable for train and val and ops
# to swap in pre-fetched image data.
def tvSwap(tset,vset):
    tbatch, vbatch = tset.batch, vset.batch
    variables, tSwap, vSwap = [], [], []
    for i in range(len(tbatch)):
        var = tf.Variable(tf.zeros(tbatch[i].shape,dtype=tbatch[i].dtype),trainable=False)
        variables.append(var)
        tSwap.append(tf.assign(var, tf.identity(tbatch[i])).op)
        vSwap.append(tf.assign(var, tf.identity(vbatch[i])).op)
    return variables, tf.group(tSwap), tf.group(vSwap)
