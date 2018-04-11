from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np
from random import *

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
colour_inv_map = {v: k for k, v in colour_map.items()}
def deprocess(x):
    return np.uint8((x+1)/2*255)

def make_latent_samples(n_samples, sample_size):
    # return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

def save_imgs(generator,col):
    r, c = 2, 5
    x=colour_inv_map[col]
    noise = np.random.normal(0, 1, (r * c, 100))

    sampled_labels = np.full([10,], x,dtype=int).reshape(-1, 1)

    gen_imgs = generator.predict([noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    # fig.suptitle("ACGAN: generated color:%s"%colour_map[x], fontsize=12)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:],cmap='jet')
            # axs[i,j].set_title("Color:%s " % colour_map[int(sampled_labels[cnt])])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("%s.png"%col)
    plt.close()

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
    json_file = open('/Users/shivendra/Desktop/CU/HCML/Logo_generation_GAN/saved_model/vanilla_dcgan_generator.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/Users/shivendra/Desktop/CU/HCML/Logo_generation_GAN/saved_model/vanilla_dcgan_generator_weights.hdf5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')
    return loaded_model

def give_me_color():
    optimizer = Adam(0.0002, 0.5)
    json_file = open('C:\\Users\Shivendra\Desktop\GAN\GAN_HCML\saved_model\\acgan_generator.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:\\Users\Shivendra\Desktop\GAN\GAN_HCML\saved_model\\acgan_generator_weights.hdf5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=['binary_crossentropy'],
                               optimizer=optimizer)
    return loaded_model


# generator=gimme_something()
# show_images(generator.predict(make_latent_samples(25, 100)))


if __name__ == "__main__":
    generator=gimme_something()
    show_images(generator.predict(make_latent_samples(25, 100)))
    generator_color = give_me_color()
    save_imgs(generator_color, 'red')
