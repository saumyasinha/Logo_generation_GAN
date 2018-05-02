"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.ops.lld
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
X=np.load('C:\\Users\Shivendra\Desktop\GAN\icon_sharp_dataset_from_hd5.npy')
y=np.load('C:\\Users\Shivendra\Desktop\GAN\\rc_cluster_icon_sharp_dataset_from_hd5.npy')
print(np.unique(y))
print(X.shape,y.shape)
X=X[:y.shape[0]]
split = int(0.8 * len(X))
n=X.shape[0]
X = X.reshape(n,-1)
print(X.shape,y.shape)
X_train, X_test = X[:split, ], X[split:, ]
y_train, y_test = y[:split, ], y[split:, ]

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 20000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    print('true')
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=16)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=16)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, int(4*4*DIM_G), noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 16, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data_splits.append(Generator(int(BATCH_SIZE/len(DEVICES)), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:int(len(DEVICES)/2)]
    DEVICES_A = DEVICES[int(len(DEVICES)/2):]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i],
                all_real_data_splits[len(DEVICES_A)+i],
                fake_data_splits[i],
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
            disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]
            disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
            if CONDITIONAL and ACGAN:
                disc_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:int(BATCH_SIZE/len(DEVICES_A))], labels=real_and_fake_labels[:int(BATCH_SIZE/len(DEVICES_A))])
                ))
                disc_acgan_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[:int(BATCH_SIZE/len(DEVICES_A))], dimension=1)),
                            real_and_fake_labels[:int(BATCH_SIZE/len(DEVICES_A))]
                        ),
                        tf.float32
                    )
                ))
                disc_acgan_fake_accs.append(tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.to_int32(tf.argmax(disc_all_acgan[int(BATCH_SIZE/len(DEVICES_A)):], dimension=1)),
                            real_and_fake_labels[int(BATCH_SIZE/len(DEVICES_A)):]
                        ),
                        tf.float32
                    )
                ))


    for i, device in enumerate(DEVICES_B):
        with tf.device(device):
            real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
            fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
            labels = tf.concat([
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i],
            ], axis=0)
            alpha = tf.random_uniform(
                shape=[int(BATCH_SIZE/len(DEVICES_A)),1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
            disc_costs.append(gradient_penalty)

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
    if CONDITIONAL and ACGAN:
        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
        disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
        disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = GEN_BS_MULTIPLE * int(BATCH_SIZE / len(DEVICES))
            fake_labels = tf.cast(tf.random_uniform([n_samples])*16, tf.int32)
            if CONDITIONAL and ACGAN:
                disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                ))
            else:
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
    if CONDITIONAL and ACGAN:
        gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))


    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    # frame_i = [0]
    # fixed_noise = tf.constant(np.random.normal(size=(80, 128)).astype('float32'))
    # fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15] * 5, dtype='int32'))
    # fixed_noise_samples = Generator(80, fixed_labels, noise=fixed_noise)
    #
    #
    # def generate_image(frame, true_dist):
    #     samples = session.run(fixed_noise_samples)
    #     samples = ((samples + 1.) * (255. / 2)).astype('int32')
    #     lib.save_images.save_images(samples.reshape((80, 3, 32, 32)),
    #                                 'C:\\Users\Shivendra\Desktop\GAN\iwgan_acgan_16rc_icon_sharp\samples_{}.png'.format(
    #                                     frame))
    #
    #
    # # Function for calculating inception score
    # fake_labels_100 = tf.cast(tf.random_uniform([100])*16, tf.int32)
    # # fake_labels_100.eval()
    # samples_100 = Generator(100, fake_labels_100)
    # def get_inception_score(n):
    #     all_samples = []
    #     for i in range(int(n/100)):
    #         all_samples.append(session.run(samples_100))
    #     all_samples = np.concatenate(all_samples, axis=0)
    #     all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
    #     all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    #     return lib.inception_score.get_inception_score(list(all_samples))
    #
    # train_gen, dev_gen = lib.ops.lld.load(BATCH_SIZE, X_train, X_test, y_train, y_test)
    # def inf_train_gen():
    #     while True:
    #         for images,_labels in train_gen():
    #             yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print("\t{} ({})".format(v.name, shape_str))
        print("Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        ))
    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    saver.restore(session, "/tmp/model_ac_16rc.ckpt-24999")
    print("model restored")

    def save_images(cluster):
      frame_i = [0]
      fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
      fixed_labels = tf.constant(np.array([cluster] * 100, dtype='int32'))
      fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
      frame = cluster
      samples = session.run(fixed_noise_samples)
      samples = ((samples + 1.) * (255. / 2)).astype('int32')
      lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                  'C:\\Users\Shivendra\Desktop\GAN\iwgan_acgan_16rc_lld_saved_model\samples_{}.png'.format(
                                      frame))




    # saver.restore(session, tf.train.latest_checkpoint('./'))


    for i in range(16):
        save_images(i)

# import numpy as np
# import random
#
# X=np.load('C:\\Users\Shivendra\Desktop\GAN\icon_sharp_dataset_from_hd5.npy')
# y=np.load('C:\\Users\Shivendra\Desktop\GAN\\rc_cluster_icon_sharp_dataset_from_hd5.npy')
# # print(np.unique(y))
# X=X[:y.shape[0]]
# uniq_labels=np.unique(y)
#
# for i in uniq_labels:
#     ix = np.isin(y, [i])
#
#     temp=X[np.where(ix)]
#     indices = np.random.randint(0, temp.shape[0], 100)
#     samples = temp[indices]
#     print(samples.shape)
#     # samples=temp[:100]
#     lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
#                                 'C:\\Users\Shivendra\Desktop\GAN\iwgan_acgan_16rc_lld_original\samples_{}.png'.format(
#                                     i))