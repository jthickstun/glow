#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time

import cv2
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import graphics
from utils import ResultLogger

import tfops as Z
from model import prior

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init(hps, iterator, sess):
    xl = []
    #for i in range(hps.n_batch_test):
    for i in range(1):
        if hps.direct_iterator:
            x, y = sess.run(iterator.get_next())
        else:
            x, y = iterator()
        xl.append(x)

    x = np.concatenate(xl,axis=0)

#    return 1.0*np.random.randn(*x.shape)
    return np.random.uniform(0,1,x.shape)


def write_images(x,name,n=7):
    d = x.shape[1]
    panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j]+.5)).clip(0,255).astype(np.uint8)[:,:,::-1]

    cv2.imwrite('sample_output/' + name, panel)


# ===
# Code for getting data
# ===
def get_data(hps, sess):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300*hvd.size(), 'lsun': 300*hvd.size()}[hps.problem]
    hps.n_y = {'mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': 'data/celeba-tfr', 'lsun_realnvp': '/mnt/hdd2/thickstn/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]
        if hps.category:
            hps.data_dir += ('/%s' % hps.category)

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def main(hps):

    hps.flow_coupling = 1
    
    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    x = init(hps, test_iterator, sess) 

    hps.inference = True
    checkpoints = ['logs1point0','logs0point599','logs0point359','logs0point215','logs0point129','logs0point077','logs0point046','logs0point027','logs0point016','logs0point01','logs0point0']
    sigmas = np.array([1.0, 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.027, 0.01668101, 0.01, 0.001])
    iteration = 0
    for sigma,checkpoint in zip(sigmas,checkpoints):
        #hps.restore_path = 'celebalogs/{}/model_best_loss.ckpt'.format(checkpoint)
        hps.restore_path = 'cifar10logs/{}/model_best_loss.ckpt'.format(checkpoint)
        print("Using checkpoint {}".format(hps.restore_path))

        # Create model
        import model
        train_iterator, test_iterator, data_init = get_data(hps, sess)
        curr_model = model.model(sess, hps, train_iterator, test_iterator, data_init, train=False)

        hps.noise_level = sigma
        estimate_nll(sess, curr_model, hps, test_iterator)
        hps.noise_level = 0

        y = np.zeros([hps.n_batch_test], dtype=np.int32)

        eta = .000005 * (sigma / .01) ** 2
        x += sigma*np.random.randn(*x.shape)
        for i in range(200):
            grad_x = curr_model.grad_logprob(x,y)

            logprob = curr_model.logprob(x,y).mean()
            normx = np.linalg.norm(grad_x.reshape(-1,3*32*32),axis=1).mean()
            print('logpx: {}, normx: {}, eta: {}'.format(logprob/(32*32*3*np.log(2)),normx, eta))

            epsilon = np.sqrt(2*eta)*np.random.randn(*(x.shape))
            x = x + eta * grad_x + epsilon
            iteration += 1

        write_images(x, 'x{}.png'.format(sigma))

        tf.reset_default_graph()
        sess = tensorflow_session()

def estimate_nll(sess, model, hps, iterator):
    if hps.direct_iterator:
        iterator = iterator.get_next()

    print('Running inference on {} data points'.format(hps.full_test_its*hps.n_batch_test))
    logpz = []
    grad_logpz = []
    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            x, y = sess.run(iterator)
        else:
            x, y = iterator()

        # preprocess
        x = x/256. - .5
        x += np.random.randn(*(x.shape)) * hps.noise_level

        logpz.append(model.logprob(x,y))

    logpz = np.concatenate(logpz, axis=0)
    print('NLL = {}'.format(-logpz.mean()/(32*32*3*np.log(2))))


# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    parser.add_argument("--noise_level", type=float, default=0,
                        help="Amount of noise to add")

    hps = parser.parse_args()  # So error if typo
    main(hps)
