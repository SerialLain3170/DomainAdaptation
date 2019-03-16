import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers
from pathlib import Path
import argparse
from model import Encoder, Classification
from scipy.io import loadmat
import numpy as np

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model):
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    return optimizer

parser = argparse.ArgumentParser(description="Pretrain Souce CNN")
parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
parser.add_argument('--b', type=int, default=100, help="batch size")

args = parser.parse_args()
epochs = args.e
batchsize = args.b

encoder = Encoder(in_ch=3)
encoder.to_gpu()
enc_opt = set_optimizer(encoder)

classification = Classification()
classification.to_gpu()
class_opt = set_optimizer(classification)

x = loadmat("./Dataset/train_32x32.mat")['X']
t = loadmat("./train_32x32.mat")['y']
x_test = loadmat("./Dataset/test_32x32.mat")['X']
t_test = loadmat("./Dataset/test_32x32.mat")['y']

Ntrain = x.shape[3]
Ntest = 26000

for index in range(Ntest):
    if t_test[index] == 10:
        t_test[index] = 0

test = x_test.transpose(3, 2,0,1).astype(np.float32)
test = test / 255.0
test = chainer.as_variable(xp.array(test).astype(xp.float32))

for epoch in range(epochs):
    sum_loss = 0
    for batch in range(0, Ntrain, batchsize):
        svhn_box = []
        label_box = []
        for _ in range(batchsize):
            rnd = np.random.randint(Ntrain)
            img = x[:,:,:,rnd]
            img = img.transpose(2,0,1).astype(np.float32)
            img = img / 255.0
            svhn_box.append(img)

            label = t[rnd]
            if label == [10]:
                label = [0]
            label_box.append(label)

        s = chainer.as_variable(xp.array(svhn_box).astype(xp.float32))
        l = chainer.as_variable(xp.array(label_box).astype(xp.int32))[:, 0]

        y = classification(encoder(s))
        loss = F.softmax_cross_entropy(y, l)

        encoder.cleargrads()
        classification.cleargrads()

        loss.backward()

        enc_opt.update()
        class_opt.update()

        sum_loss += loss.data.get()

        if epoch % 10 == 0 and batch == 0:
            serializers.save_npz('encoder.model', encoder)
            serializers.save_npz('classification.model', classification)
            
            count = 0
            for b in range(0, Ntest, batchsize):
                test_batch = test[b : b+batchsize]
                with chainer.using_config('train', False):
                    y = F.softmax(classification(encoder(test_batch)))
                y.unchain_backward()
                y = y.data.get()
                for index in range(b, b+batchsize):
                    if t_test[index][0] == np.argmax(y[index - b]):
                        count += 1

            print('accuracy : {}'.format(count / Ntest))
    print('epoch : {} loss : {}'.format(epoch, sum_loss/Ntrain))