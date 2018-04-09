import numpy as np
from scipy.io import loadmat
import keras
import keras.backend as K
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def preprocess(x):
    return (x/255)*2-1

def deprocess(x):
    return np.uint8((x+1)/2*255)

class DCGAN:
    '''
    CNN classifier
    '''
    def __init__(self, X_train_real, X_test_real):

        '''
        Initialize CNN classifier data
        '''
        # self.batch_size = batch_size
        # self.epochs = epochs
        self.X_train_real=X_train_real
        self.X_test_real = X_test_real

    def make_generator(self,input_size, leaky_alpha):
        # generates images in (32,32,3)
        return Sequential([
            Dense(4 * 4 * 512, input_shape=(input_size,)),
            Reshape(target_shape=(4, 4, 512)),  # 4,4,512
            BatchNormalization(),
            LeakyReLU(alpha=leaky_alpha),
            Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'),  # 8,8,256
            BatchNormalization(),
            LeakyReLU(alpha=leaky_alpha),
            Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'),  # 16,16,128
            BatchNormalization(),
            LeakyReLU(alpha=leaky_alpha),
            Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'),  # 32,32,3
            Activation('tanh')
        ])

    def make_discriminator(self,leaky_alpha):
        # classifies images in (32,32,3)
        return Sequential([
            Conv2D(64, kernel_size=5, strides=2, padding='same',  # 16,16,64
                   input_shape=(32, 32, 3)),
            LeakyReLU(alpha=leaky_alpha),
            Conv2D(128, kernel_size=5, strides=2, padding='same'),  # 8,8,128
            BatchNormalization(),
            LeakyReLU(alpha=leaky_alpha),
            Conv2D(256, kernel_size=5, strides=2, padding='same'),  # 4,4,256
            BatchNormalization(),
            LeakyReLU(alpha=leaky_alpha),
            Flatten(),
            Dense(1),
            Activation('sigmoid')
        ])

    # beta_1 is the exponential decay rate for the 1st moment estimates in Adam optimizer
    def make_DCGAN(self, sample_size, g_learning_rate, g_beta_1, d_learning_rate, d_beta_1,leaky_alpha):
        print('dong')
        # clear any session data
        K.clear_session()

        # generator
        generator = self.make_generator(sample_size, leaky_alpha)

        # discriminator
        discriminator = self.make_discriminator(leaky_alpha)
        discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')

        # GAN
        gan = Sequential([generator, discriminator])
        gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')

        return gan, generator, discriminator

    def make_latent_samples(self,n_samples, sample_size):
        # return np.random.uniform(-1, 1, size=(n_samples, sample_size))
        return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

    def make_trainable(self,model, trainable):
        for layer in model.layers:
            layer.trainable = trainable

    def make_labels(self,size):
        print("ding")
        return np.ones([size, 1]), np.zeros([size, 1])

    def show_losses(self,losses):
        losses = np.array(losses)

        fig, ax = plt.subplots()
        plt.plot(losses.T[0], label='Discriminator')
        plt.plot(losses.T[1], label='Generator')
        plt.title("Validation Losses")
        plt.legend()
        plt.savefig('loss.png')

    def show_images(self,generated_images):
        n_images = len(generated_images)
        print(n_images)
        cols = 10
        rows = n_images // cols

        plt.figure()
        for i in range(n_images):
            img = deprocess(generated_images[i])
            ax = plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        # plt.show()
        plt.savefig('generated_images.png')
        plt.close('all')

    def train(self,
            g_learning_rate,  # learning rate for the generator
            g_beta_1,  # the exponential decay rate for the 1st moment estimates in Adam optimizer
            d_learning_rate,  # learning rate for the discriminator
            d_beta_1,  # the exponential decay rate for the 1st moment estimates in Adam optimizer
            leaky_alpha,
            smooth=0.1,
            sample_size=100,  # latent sample size (i.e. 100 random numbers)
            epochs=20,
            batch_size=16,  # train batch size
            eval_size=10,  # evaluate size
            show_details=True):
        print("reached here")
        # labels for the batch size and the test size
        y_train_real, y_train_fake = self.make_labels(batch_size)
        y_eval_real, y_eval_fake = self.make_labels(eval_size)

        # create a GAN, a generator and a discriminator
        gan, generator, discriminator = self.make_DCGAN(
            sample_size,
            g_learning_rate,
            g_beta_1,
            d_learning_rate,
            d_beta_1,
            leaky_alpha)
        print(epochs)
        losses = []
        for e in range(epochs):
            for i in range(len(X_train_real) // batch_size):
                # real SVHN digit images
                # print(i)
                X_batch_real = X_train_real[i * batch_size:(i + 1) * batch_size]

                # latent samples and the generated digit images
                latent_samples = self.make_latent_samples(batch_size, sample_size)
                X_batch_fake = generator.predict_on_batch(latent_samples)

                # train the discriminator to detect real and fake images
                self.make_trainable(discriminator, True)
                discriminator.train_on_batch(X_batch_real, y_train_real * (1 - smooth))
                discriminator.train_on_batch(X_batch_fake, y_train_fake)

                # train the generator via GAN
                self.make_trainable(discriminator, False)
                #discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')
                gan.train_on_batch(latent_samples, y_train_real)

            # evaluate
            X_eval_real = X_test_real[np.random.choice(len(X_test_real), eval_size, replace=False)]

            latent_samples = self.make_latent_samples(eval_size, sample_size)
            X_eval_fake = generator.predict_on_batch(latent_samples)

            d_loss = discriminator.test_on_batch(X_eval_real, y_eval_real)
            d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)
            g_loss = gan.test_on_batch(latent_samples, y_eval_real)  # we want the fake to be realistic!

            losses.append((d_loss, g_loss))

            print("Epoch: {:>3}/{} Discriminator Loss: {:>7.4f}  Generator Loss: {:>7.4f}".format(
                e + 1, epochs, d_loss, g_loss))

            # show the generated images
            if (e + 1) % 5 == 0:
                self.show_images(X_eval_fake[:10])

        if show_details:
            self.show_losses(losses)
            self.show_images(generator.predict(self.make_latent_samples(80, sample_size)))
        return generator




if __name__ == '__main__':
    X = np.load('C:\\Users\\Shivendra\\Desktop\\GAN\\icon_dataset.npy')
    X=X[:20000]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split, ], X[split:, ]
    X_train_real = preprocess(X_train)
    X_test_real = preprocess(X_test)

    print(X_train_real.shape)
    print(X_test_real.shape)

    cnn = DCGAN(X_train_real, X_test_real)

    # sample_images = X_train_real[np.random.choice(len(X_train), size=80, replace=False)]
    #
    # plt.figure(figsize=(10, 8))
    # for i in range(80):
    #     plt.subplot(8, 10, i+1)
    #     plt.imshow(sample_images[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout()
    # plt.show()

    cnn.train(g_learning_rate=0.0001, g_beta_1=0.5, d_learning_rate=0.001, d_beta_1=0.5, leaky_alpha=0.2)
