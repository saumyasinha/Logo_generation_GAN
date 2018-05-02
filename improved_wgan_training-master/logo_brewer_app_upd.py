
"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())
# import matplotlib.pyplot as plt
import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.ops.lld
import tflib.inception_score
import tflib.plot

# from scipy.misc import imsave
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
# X=np.load('C:\\Users\Shivendra\Desktop\GAN\icon_dataset_from_hd5.npy')
# y=np.load('C:\\Users\Shivendra\Desktop\GAN\\rc_cluster_icon_dataset_from_hd5.npy')
# print(np.unique(y))
# X=X[:y.shape[0]]
# ix = np.isin(y, [0,1,2,3,4,5,6,7,8,9])
# X_subset=X[np.where(ix)]
# y_subset=y[np.where(ix)]
# print(np.unique(y_subset))
# split = int(0.8 * len(X_subset))
# n=X_subset.shape[0]
# X_subset = X_subset.reshape(n,-1)
# print(X_subset.shape,y_subset.shape)
# X_train, X_test = X_subset[:split, ], X_subset[split:, ]
# y_train, y_test = y_subset[:split, ], y_subset[split:, ]
def gimme_something(saved_model,image_label):
    MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
    DIM = 64  # This overfits substantially; you're probably better off with 64
    LAMBDA = 10  # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = 5  # How many critic iterations per generator iteration
    BATCH_SIZE = 16  # Batch size
    ITERS = 50000  # How many generator iterations to train for
    OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)

    lib.print_model_settings(locals().copy())

    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha * x, x)

    def ReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs)
        return tf.nn.relu(output)

    def LeakyReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs)
        return LeakyReLU(output)

    def Generator(n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * DIM, noise)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4 * DIM, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    def Discriminator(inputs):
        output = tf.reshape(inputs, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2 * DIM, 5, output, stride=2)
        if MODE != 'wgan-gp':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5, output, stride=2)
        if MODE != 'wgan-gp':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * DIM, 1, output)

        return tf.reshape(output, [-1])

    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_data = 2 * ((tf.cast(real_data_int, tf.float32) / 255.) - .5)
    fake_data = Generator(BATCH_SIZE)

    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

        clip_ops = []
        for var in disc_params:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                 var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=disc_params)

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
        disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
        disc_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name(
                                                                                          'Generator'))
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name(
                                                                                           'Discriminator.'))
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

    saver.restore(session, saved_model)
    print("model restored")

    # For generating samples
    fixed_noise_128 = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_noise_samples_128 = Generator(100, noise=fixed_noise_128)

    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples_128)
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                    'C:\\Users\Shivendra\Desktop\GAN\logo_brewer_app_images\samples_{}.png'.format(
                                                image_label))


def gimme_labels(saved_model,image_label,cluster=None):
    print(cluster)
    N_GPUS = 1
    if N_GPUS not in [1,2]:
        raise Exception('Only 1 or 2 GPUs supported!')

    BATCH_SIZE = 64 # Critic batch size
    GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
    ITERS = 25000 # How many iterations to train for
    DIM_G = 128 # Generator dimensionality
    DIM_D = 128 # Critic dimensionality
    NORMALIZATION_G = True # Use batc hnorm in generator?
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
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
        elif ('Generator' in name) and NORMALIZATION_G:
            if labels is not None:
                return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
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
            output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
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
                fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
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

        # session.run(tf.initialize_all_variables())
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        saver.restore(session, saved_model)
        print("model restored")

        def generate_images(cluster):
            if cluster==None:
                frame_i = [0]
                fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
                fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
                fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
                samples = session.run(fixed_noise_samples)
                samples = ((samples + 1.) * (255. / 2)).astype('int32')
                lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                            'C:\\Users\Shivendra\Desktop\GAN\logo_brewer_app_images\samples_{}.png'.format(
                                                image_label))

            else:
                frame_i = [0]
                fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
                fixed_labels = tf.constant(np.array([cluster] * 100, dtype='int32'))
                fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
                samples = session.run(fixed_noise_samples)
                samples = ((samples + 1.) * (255. / 2)).astype('int32')
                lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                            'C:\\Users\Shivendra\Desktop\GAN\logo_brewer_app_images\samples_of_cluster{}.png'.format(
                                                cluster))
        # scipy.misc.imsave(image_label, samples.reshape((100, 3, 32, 32)))


        generate_images(cluster)

if __name__ == "__main__":
    saved_model="/tmp/model_ac_colo.ckpt-4999"
    # generator=gimme_labels(saved_model, "RC_cluster_labels")
    gimme_labels(saved_model, "color_cluster_labels")
    gimme_something(saved_model,"vanilla_iWGAN")


