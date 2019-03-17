import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import cv2 as cv
import argparse
import numpy as np
from scipy.io import loadmat
from chainer import cuda,initializers,optimizers,serializers,datasets
from model import Classification, Encoder

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002):
    optimizer = optimizers.Adam(alpha=alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    return optimizer

def prepare_dataset(index):
    img = cv.resize(test_mnist[index][0].transpose(1,2,0), (32, 32))
    img = img.transpose(2,0,1).astype(np.float32)

    return img

def discrepancy_loss(m1, m2):
    return F.mean_absolute_error(F.softmax(m1), F.softmax(m2))

parser = argparse.ArgumentParser(description="Train")
parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
parser.add_argument('--b', type=int, default=250, help="batch size")
parser.add_argument('--n', type=int, default=4, help="the repeat number of stepC")

args = parser.parse_args()
epochs = args.e
batchsize = args.b
n_stepc = args.n

generator = Encoder(in_ch=3)
generator.to_gpu()
gen_opt = set_optimizer(generator)

classifier_1 = Classification()
classifier_1.to_gpu()
c1_opt = set_optimizer(classifier_1)

classifier_2 = Classification()
classifier_2.to_gpu()
c2_opt = set_optimizer(classifier_2)

train_svhn = loadmat('./Dataset/train_32x32.mat')['X']
label_svhn = loadmat('./Dataset/train_32x32.mat')['y']
train_mnist, test_mnist = datasets.get_mnist(ndim=3, rgb_format=True)

Nsvhn = train_svhn.shape[3]
Nmnist = 60000
Ntest = 10000

for epoch in range(epochs):
    sum_stepA_loss = 0
    sum_stepB_loss = 0
    sum_stepC_loss = 0
    for batch in range(0, Nsvhn, batchsize):
        svhn_box = []
        label_box = []
        mnist_box = []
        for _ in range(batchsize):
            rnd = np.random.randint(Nsvhn)
            img = train_svhn[:, :, :, rnd]
            img = img.transpose(2,0,1).astype(np.float32)
            img = img / 255.0
            svhn_box.append(img)

            label = label_svhn[rnd][0]
            if label == 10:
                label = 0
            label_box.append(label)

            rnd = np.random.randint(Nmnist)
            img = cv.resize(train_mnist[rnd][0].transpose(1,2,0), (32, 32))
            img = img.transpose(2,0,1).astype(np.float32)
            mnist_box.append(img)

        s = chainer.as_variable(xp.array(svhn_box).astype(xp.float32))
        m = chainer.as_variable(xp.array(mnist_box).astype(xp.float32))
        t = chainer.as_variable(xp.array(label_box).astype(xp.int32))

        # stepA
        y = generator(s)
        s1 = classifier_1(y)
        s2 = classifier_2(y)
        loss = F.softmax_cross_entropy(s1, t)
        loss += F.softmax_cross_entropy(s2, t)

        generator.cleargrads()
        classifier_1.cleargrads()
        classifier_2.cleargrads()
        loss.backward()
        gen_opt.update()
        c1_opt.update()
        c2_opt.update()
        loss.unchain_backward()

        sum_stepA_loss += loss.data.get()

        # stepB
        y_s = generator(s)
        y_t = generator(m)
        s1 = classifier_1(y_s)
        s2 = classifier_2(y_s)
        m1 = classifier_1(y_t)
        m2 = classifier_2(y_t)

        y_s.unchain_backward()
        y_t.unchain_backward()

        loss = F.softmax_cross_entropy(s1, t)
        loss += F.softmax_cross_entropy(s2, t)
        loss -= discrepancy_loss(m1, m2)

        classifier_1.cleargrads()
        classifier_2.cleargrads()
        loss.backward()
        c1_opt.update()
        c2_opt.update()
        loss.unchain_backward()

        sum_stepB_loss += loss.data.get()

        # stepC
        classifier_1.disable_update()
        classifier_2.disable_update()
        for _ in range(n_stepc):
            y_t = generator(m)
            m1 = classifier_1(y_t)
            m2 = classifier_2(y_t)

            loss = discrepancy_loss(m1, m2)

            generator.cleargrads()
            classifier_1.cleargrads()
            classifier_2.cleargrads()
            loss.backward()
            gen_opt.update()
            c1_opt.update()
            c2_opt.update()
            loss.unchain_backward()

            sum_stepC_loss += loss.data.get()

        classifier_1.enable_update()
        classifier_2.enable_update()

        if epoch % 10 == 0 and batch == 0:
            serializers.save_npz('generator.model', generator)
            serializers.save_npz('classifier_1.model', classifier_1)
            serializers.save_npz('classifier_2.model', classifier_2)

            count_1 = 0
            count_2 = 0
            count_ens = 0
            for b in range(0, Ntest, batchsize):
                test_batch = []
                label_batch = []
                for index in range(batchsize):
                    img = prepare_dataset(b + index)
                    test_batch.append(img)
                    label_batch.append(test_mnist[b+index][1])
                test_batch = chainer.as_variable(xp.array(test_batch).astype(xp.float32))
                with chainer.using_config('train', False):
                    y = generator(test_batch)
                    y1 = classifier_1(y)
                    y2 = classifier_2(y)
                    yens = y1 + y2
                    y1, y2, yens = F.softmax(y1), F.softmax(y2), F.softmax(yens)
                y1.unchain_backward()
                y2.unchain_backward()
                yens.unchain_backward()
                y1 = y1.data.get()
                y2 = y2.data.get()
                yens = yens.data.get()

                for index in range(batchsize):
                    if label_batch[index] == np.argmax(y1[index]):
                        count_1 += 1
                    if label_batch[index] == np.argmax(y2[index]):
                        count_2 += 1
                    if label_batch[index] == np.argmax(yens[index]):
                        count_ens += 1

            print('accuracy_1 : {}'.format(count_1 / Ntest))
            print('accuracy_2 : {}'.format(count_2 / Ntest))
            print('accuracy_ens : {}'.format(count_ens / Ntest))

    print('epoch : {}'.format(epoch))
    print('stepA loss :{}'.format(sum_stepA_loss / Nsvhn))
    print('stepB loss :{}'.format(sum_stepB_loss / Nsvhn))
    print('stepC loss :{}'.format(sum_stepC_loss / Nsvhn))