from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np

def deprocess(x):
    return np.uint8((x+1)/2*255)

def make_latent_samples(n_samples, sample_size):
    # return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))


def show_images(generated_images):
    n_images = len(generated_images)
    print(n_images)
    cols = 5
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
    plt.savefig('generated_images_for_logo_brewer.png')
    plt.close('all')

def gimme_something():
    g_learning_rate = 0.0001
    g_beta_1 = 0.5
    json_file = open('C:\\Users\Shivendra\Desktop\GAN\GAN_HCML\saved_model\\vanilla_dcgan_generator.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:\\Users\Shivendra\Desktop\GAN\GAN_HCML\saved_model\\vanilla_dcgan_generator_weights.hdf5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')
    return loaded_model

generator=gimme_something()
show_images(generator.predict(make_latent_samples(25, 100)))
