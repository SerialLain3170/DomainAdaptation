import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import cv2 as cv
import argparse
import numpy as np
import pylab
import os
from scipy.io import loadmat
from chainer import cuda,initializers,optimizers,serializers,datasets
from model import Classification, Encoder, Generator, Discriminator, TaskNet, Discriminator_feature

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

parser = argparse.ArgumentParser(description="Train")
parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
parser.add_argument('--b', type=int, default=100, help="batch size")
parser.add_argument('--n', type=int, default=4, help="the repeat number of stepC")
parser.add_argument('--testsize', type=int, default=8, help="test size")

args = parser.parse_args()
epochs = args.e
batchsize = args.b
n_stepc = args.n
testsize = args.testsize

image_dir = './output/'
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

generator_st = Generator()
generator_st.to_gpu()
gen_st_opt = set_optimizer(generator_st)

generator_ts = Generator()
generator_ts.to_gpu()
gen_ts_opt = set_optimizer(generator_ts)

discriminator_st = Discriminator()
discriminator_st.to_gpu()
dis_st_opt = set_optimizer(discriminator_st)

discriminator_ts = Discriminator()
discriminator_ts.to_gpu()
dis_ts_opt = set_optimizer(discriminator_ts)

discriminator_feat = Discriminator_feature()
discriminator_feat.to_gpu()
dis_feat_opt = set_optimizer(discriminator_feat)

source_tasknet = TaskNet()
source_tasknet.to_gpu()
s_task_opt = set_optimizer(source_tasknet)
serializers.load_npz('./taksnet.model', source_tasknet)
source_tasknet.disable_update()

target_tasknet = TaskNet()
target_tasknet.to_gpu()
t_task_opt = set_optimizer(target_tasknet)
target_tasknet.copyparams(source_tasknet)

train_svhn = loadmat('./Dataset/train_32x32.mat')['X']
label_svhn = loadmat('./Dataset/train_32x32.mat')['y']
test_svhn = loadmat('./Dataset/test_32x32.mat')['X']
train_mnist, test_mnist = datasets.get_mnist(ndim=3, rgb_format=True)

test_svhn = test_svhn[:, :, :, -8:]
test = test_svhn.transpose(3,2,0,1).astype(np.float32)
test /= 255.0
test = chainer.as_variable(xp.array(test).astype(xp.float32))

Nsvhn = train_svhn.shape[3]
Nmnist = 60000
Ntest = 10000

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    sum_sem_loss = 0
    sum_feat_loss = 0
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
        t = chainer.as_variable(xp.array(mnist_box).astype(xp.float32))
        l = chainer.as_variable(xp.array(label_box).astype(xp.int32))

        s_t = generator_st(s)
        t_s = generator_ts(t)

        dis_t_fake = discriminator_st(s_t)
        dis_t_real = discriminator_st(t)
        dis_s_fake = discriminator_ts(t_s)
        dis_s_real = discriminator_ts(s)

        dis_t_loss = F.mean(F.softplus(-dis_t_real)) + F.mean(F.softplus(dis_t_fake))
        dis_s_loss = F.mean(F.softplus(-dis_s_real)) + F.mean(F.softplus(dis_s_fake))
        dis_loss = dis_t_loss + dis_s_loss

        s_t_encode = source_tasknet(s_t)
        t_encode = target_tasknet(t)

        dis_feat_real = discriminator_feat(s_t_encode)
        dis_feat_fake = discriminator_feat(t_encode)
        dis_feat_loss = F.mean(F.softplus(-dis_feat_real)) + F.mean(F.softplus(dis_feat_fake))
        dis_loss += dis_feat_loss

        discriminator_st.cleargrads()
        discriminator_ts.cleargrads()
        discriminator_feat.cleargrads()
        dis_loss.backward()
        dis_st_opt.update()
        dis_ts_opt.update()
        dis_feat_opt.update()
        dis_loss.unchain_backward()

        # CycleGAN Adversarial loss
        s_t = generator_st(s)
        t_s = generator_ts(t)

        dis_t_fake = discriminator_st(s_t)
        dis_s_fake = discriminator_ts(t_s)
        gen_loss = F.mean(F.softplus(-dis_t_fake)) + F.mean(F.softplus(-dis_s_fake))

        # Cycle-consistency loss
        s_t_s = generator_ts(s_t)
        t_s_t = generator_st(t_s)

        cycle_s_loss = F.mean_absolute_error(s, s_t_s)
        cycle_t_loss = F.mean_absolute_error(t, t_s_t)
        cycle_loss = cycle_s_loss + cycle_t_loss

        # Target TaskNet loss
        s_t_encode = target_tasknet(s_t)
        loss = F.softmax_cross_entropy(s_t_encode, l)

        # Semantic-consistency loss
        y_s = xp.argmax(F.softmax(source_tasknet(s)).data, axis=1)
        y_s = chainer.as_variable(y_s)
        y_t = xp.argmax(F.softmax(source_tasknet(t)).data, axis=1)
        y_t = chainer.as_variable(y_t)

        ss_t_encode = source_tasknet(s_t)
        sem_t_loss = F.softmax_cross_entropy(ss_t_encode, y_s)

        t_s_encode = source_tasknet(t_s)
        sem_s_loss = F.softmax_cross_entropy(t_s_encode, y_t)
        sem_loss = sem_t_loss + sem_s_loss

        # Feature-level loss
        t_encode = target_tasknet(t)
        dis_feat_fake = discriminator_feat(t_encode)
        gen_feat_loss = F.mean(F.softplus(-dis_feat_fake))

        gen_loss += loss + cycle_loss + sem_loss + gen_feat_loss

        generator_st.cleargrads()
        generator_ts.cleargrads()
        target_tasknet.cleargrads()
        gen_loss.backward()
        gen_st_opt.update()
        gen_ts_opt.update()
        t_task_opt.update()
        gen_loss.unchain_backward()

        sum_gen_loss += gen_loss.data.get()
        sum_dis_loss += dis_loss.data.get()
        sum_feat_loss += loss.data.get()
        sum_sem_loss += sem_loss.data.get()

        if epoch % 1 == 0 and batch == 0:
            serializers.save_npz('generator_st.model', generator_st)
            serializers.save_npz('generator_ts.model', generator_ts)
            serializers.save_npz('target_tasknet.model', target_tasknet)

            with chainer.using_config('train', False):
                y = generator_st(test)
            y = y.data.get()
            tmps = test.data.get()
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            for i_ in range(testsize):
                tmp = (np.clip((tmps[i_,:,:,:])*255.0, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,2*i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%svisualize_%d.png'%(image_dir, epoch))

                tmp = (np.clip((y[i_,:,:,:])*255.0, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(4,4,2*i_+2)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%svisualize_%d.png'%(image_dir, epoch))

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
                    y = F.softmax(target_tasknet(test_batch))
                y.unchain_backward()
                y = y.data.get()
                for index in range(batchsize):
                    if label_batch[index] == np.argmax(y[index]):
                        count += 1

            print('accuracy : {}'.format(count / Ntest))

    print('epoch : {}'.format(epoch))
    print('Genrator loss : {}'.format(sum_gen_loss / Nsvhn))
    print('Discriminator loss : {}'.format(sum_dis_loss / Nsvhn))