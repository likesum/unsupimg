import numpy as np
import tensorflow as tf


################## Dataset #######################
class Dataset:
    def graph(self):
        self.names, self.seeds = [], []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))
            self.seeds.append(tf.placeholder(tf.int32, shape=[4]))

        lefts, rights, shifts = [], [], []
        for i in range(self.bsz):
            img = tf.read_file(self.names[i])
            img = tf.image.decode_jpeg(img, channels=1)

            if self.all_blocks:
                # Random blocks for baseline
                dy, dx = tf.random_uniform([], 0, self.csz, dtype=tf.int32), tf.random_uniform([], 0, self.csz, dtype=tf.int32)
                lefts.append(img[dy:dy+self.imsz,dx:dx+self.imsz,:])
                dy, dx = tf.random_uniform([], 0, self.csz, dtype=tf.int32), tf.random_uniform([], 0, self.csz, dtype=tf.int32)
                rights.append(img[dy:dy+self.imsz,dx:dx+self.imsz,:])
                shifts.append(tf.convert_to_tensor([dy, dx]))
            else:
                # Fixed blocks for unsupervised learning
                shift = tf.contrib.stateless.stateless_random_uniform([2], self.seeds[i][:2], dtype=tf.float32)
                shift = 1 + tf.cast(tf.floor(shift*(self.csz-1.0)), tf.int32)
                dy, dx = shift[0], shift[1]
                lefts.append(img[:self.imsz,:self.imsz,:])
                rights.append(img[dy:dy+self.imsz,dx:dx+self.imsz,:])
                shifts.append(shift)

        lefts = tf.to_float(tf.stack(lefts))/255.0
        lefts.set_shape([self.bsz,self.imsz,self.imsz,1])
        rights = tf.to_float(tf.stack(rights))/255.0
        rights.set_shape([self.bsz,self.imsz,self.imsz,1])
        outs = [lefts, rights, tf.stack(shifts, axis=0), tf.stack(self.seeds, axis=0)]

        # Fetch op
        self.batch, self.fetchOp = [], []
        for i in range(len(outs)):
            var = tf.Variable(tf.zeros(outs[i].shape,dtype=outs[i].dtype),trainable=False)
            self.batch.append(var)
            self.fetchOp.append(tf.assign(var, outs[i]).op)
        self.fetchOp = tf.group(self.fetchOp)

    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            fd[self.names[i]] = self.files[idx]
            fd[self.seeds[i]] = self.random_seeds[idx]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
        return fd
        
    def __init__(self,lfile,bsz,imsz,csz,niter=0,isval=False,all_blocks=False):
        """
        Call with
           lfile = Path of file with list of image file names
           bsz = Batch size you want this to generate
           imsz = size of the image
           csz = size of the block
           niter = Resume at niterations
           isval = Running on train or val (random crops and shuffling for train)
        """

        self.bsz = bsz
        self.imsz = imsz
        self.csz = csz
        assert self.imsz % self.csz == 0
        self.isrand = not isval
        self.all_blocks = all_blocks

        # Load file list
        if type(lfile) is list:
            self.files = lfile
        else:
            self.files = [l.strip() for l in open(lfile).readlines()]
        if len(self.files) < bsz: # repeat file list if its size < bsz
            self.files = self.files * int(np.ceil(float(bsz)/len(self.files)))
        self.ndata = len(self.files)
        self.niter = niter*bsz

        self.random_seeds = np.random.RandomState(1234).randint(0, 65536, size=(self.ndata,4), dtype=np.int32)

        # Setup fetch graph
        self.graph()

        # Setup shuffling
        if self.isrand:
            self.rand = np.random.RandomState(0)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
        else:
            self.idx = np.int32(np.arange(self.ndata))
        

def tvSwap(tset,vset):
    tbatch, vbatch = tset.batch, vset.batch
    variables, tSwap, vSwap = [], [], []
    for i in range(len(tbatch)):
        var = tf.Variable(tf.zeros(tbatch[i].shape,dtype=tbatch[i].dtype),trainable=False)
        variables.append(var)
        tSwap.append(tf.assign(var, tf.identity(tbatch[i])).op)
        vSwap.append(tf.assign(var, tf.identity(vbatch[i])).op)
    return variables, tf.group(tSwap), tf.group(vSwap)
