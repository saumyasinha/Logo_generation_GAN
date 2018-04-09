from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply,concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class ACGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        # self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer)
        # The generator takes noise and the target label as input
        # and generates the corresponding logo-image of that label
        noise = Input(shape=(100,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(4 * 4 * 512, input_dim=100))
        model.add(Reshape((4, 4, 512)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(self.channels, kernel_size=5, strides=2, padding='same'))
        model.add(Activation("tanh"))


        model.summary()

        noise = Input(shape=(100,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        input = multiply([noise, label_embedding])

        img = model(input)

        return Model([noise, label], img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=16, save_interval=5):

        # Load the dataset
        X_train=np.load('C:\\Users\Shivendra\Desktop\GAN\icon_dataset.npy')
        y_train=np.load('C:\\Users\Shivendra\Desktop\GAN\icon_color_label.npy')
        print(X_train.shape)
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        y_train = y_train.reshape(-1, 1)

        half_batch = int(batch_size / 2)

        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch
        class_weights = [cw1, cw2]
        print(class_weights)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            # The colors labels that the generator tries to create and
            sampled_labels = np.random.randint(0, 10, half_batch).reshape(-1, 1)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Image labels. 0-9 if image is valid or 10 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = 10 * np.ones(half_batch).reshape(-1, 1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels], class_weight=class_weights)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=class_weights)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # color labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels],
                                                  class_weight=class_weights)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # self.save_model()
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        colour_map = {0: 'red',
                     1: 'green',
                     2: 'blue',
                     3: 'yellow',
                     4: 'orange',
                     5: 'white',
                     6: 'black',
                     7: 'pink',
                     8: 'purple',
                     9: 'gray'}
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.suptitle("ACGAN: Generated colours", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:],cmap='jet')
                axs[i,j].set_title("Color:%s " % colour_map[int(sampled_labels[cnt])])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:\\Users\Shivendra\Desktop\GAN\GAN_HCML\images_ACGAN\%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=25000, batch_size=16, save_interval=5000)