import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers, datasets
from pathlib import Path
import argparse
from model import Encoder, Classification, Discriminaor
from scipy.io import loadmat
import numpy as np
import cv2 as cv
import pylab

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0001):
    optimizer = optimizers.RMSprop(lr=alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(2e-5))

    return optimizer

def prepare_dataset(index):
    img = cv.resize(test_mnist[index][0].transpose(1,2,0), (32, 32))
    img = img.transpose(2,0,1).astype(np.float32)

    return img

parser = argparse.ArgumentParser(description="Train")
parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
parser.add_argument('--b', type=int, default=250, help="batch size")

args = parser.parse_args()
epochs = args.e
batchsize = args.b

source_encoder = Encoder(3)
source_encoder.to_gpu()
#senc_opt = set_optimizer(source_encoder)
serializers.load_npz('./encoder.model', source_encoder)
source_encoder.disable_update()

target_encoder =Encoder(3)
target_encoder.to_gpu()
tenc_opt = set_optimizer(target_encoder)
target_encoder.to_gpu()
target_encoder.copyparams(source_encoder)

classification = Classification()
classification.to_gpu()
class_opt = set_optimizer(classification)
serializers.load_npz('./classification.model', classification)
classification.disable_update()

discriminator = Discriminaor()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

x = loadmat("./Dataset/train_32x32.mat")['X']
train_mnist, test_mnist = datasets.get_mnist(ndim=3, rgb_format=True)

Nsvhn = x.shape[3]
Nmnist = 60000
Ntest = 10000

count = 0
for b in range(0, Ntest, batchsize):
    test_batch = []
    label_batch = []
    for index in range(batchsize):
        img = prepare_dataset(b + index)
        test_batch.append(img)
        label_batch.append(test_mnist[b+index][1])
    test_batch = chainer.as_variable(xp.array(test_batch).astype(xp.float32))
    with chainer.using_config('train', False):
        y = F.softmax(classification(target_encoder(test_batch)))
    y.unchain_backward()
    y = y.data.get()
    for index in range(batchsize):
        if label_batch[index] == np.argmax(y[index]):
            count += 1
            
print('init accuracy : {}'.format(count / Ntest))

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0
    for batch in range(0, Nsvhn, batchsize):
        svhn_box = []
        mnist_box = []
        for _ in range(batchsize):
            rnd = np.random.randint(Nsvhn)
            img = x[:, :, :, rnd]
            img = img.transpose(2,0,1).astype(np.float32)
            img = img / 255.0
            svhn_box.append(img)

            rnd = np.random.randint(Nmnist)
            img = cv.resize(train_mnist[rnd][0].transpose(1,2,0), (32, 32))
            img = img.transpose(2,0,1).astype(np.float32)
            mnist_box.append(img)

        s = chainer.as_variable(xp.array(svhn_box).astype(xp.float32))
        m = chainer.as_variable(xp.array(mnist_box).astype(xp.float32))

        y_s = source_encoder(s)
        y_m = target_encoder(m)

        y_s.unchain_backward()
        y_m.unchain_backward()

        dis_s = discriminator(y_s)
        dis_m = discriminator(y_m)
        dis_loss = -F.mean(F.log_softmax(dis_s)[:,0]) - F.mean(F.log_softmax(dis_m)[:,1])

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        y_m = target_encoder(m)
        dis_m = discriminator(y_m)

        gen_loss = -F.mean(F.log_softmax(dis_m)[:,0])

        #source_encoder.cleargrads()
        target_encoder.cleargrads()
        gen_loss.backward()
        #senc_opt.update()
        tenc_opt.update()
        gen_loss.unchain_backward()

        sum_gen_loss += gen_loss.data.get()
        sum_dis_loss += dis_loss.data.get()

        if epoch % 10 == 0 and batch == 0:
            serializers.save_npz('./target_encoder.model', target_encoder)
            count = 0
            for b in range(0, Ntest, batchsize):
                test_batch = []
                label_batch = []
                for index in range(batchsize):
                    img = prepare_dataset(b + index)
                    test_batch.append(img)
                    label_batch.append(test_mnist[b+index][1])
                test_batch = chainer.as_variable(xp.array(test_batch).astype(xp.float32))
                with chainer.using_config('train', False):
                    y = F.softmax(classification(target_encoder(test_batch)))
                y.unchain_backward()
                y = y.data.get()
                for index in range(batchsize):
                    if label_batch[index] == np.argmax(y[index]):
                        count += 1

            print('accuracy : {}'.format(count / Ntest))
    print('epoch : {}'.format(epoch))
    print('discriminator loss: {}'.format(sum_dis_loss / Nsvhn))
    print('generator loss: {}'.format(sum_gen_loss/Nsvhn))