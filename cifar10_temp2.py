# -*- coding: utf-8 -*-
"""
file: ACGAN - CIFAR10.py
author: Luke
de
Oliveira(lukedeo @ vaitech.io)
contributor: KnightTuYa(398225157 @ qq.com)
Consult
https: // github.com / lukedeo / keras - acgan
for MNIST version!
Consult
https: // github.com / soumith / ganhacks
for GAN trick!
I directly use Minibatch Layer Code from:
https://github.com/forcecore/Keras-GAN-Animeface-Character
Thanks for the great work!
I am still not satisfied with the generated images yet, Any suggestion is welcomed!

Impletmened by Yonggun Lee
"""
from __future__ import print_function
import os
from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image
from six.moves import range
import keras.backend as K
from keras.datasets import cifar10
from keras import layers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, BatchNormalization, Concatenate, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from keras.utils.generic_utils import Progbar
from Minibatch import MinibatchDiscrimination
import matplotlib.pyplot as plt
import keras.layers.merge as merge
from keras.layers.noise import GaussianNoise
import numpy as np
import h5py
from utils import get_image
from functools import partial
import math
from scipy.misc import imsave

np.random.seed(1337)
class_num = 2
label_dim = 2
K.set_image_dim_ordering('th')
path = "images"  # The path to store the generated images
load_weight = False
# Set True if you need to reload weight
load_epoch = 100  # Decide which epoch to reload weight, please check your file name

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 3, 32, 32)
    cnn = Sequential()
    cnn.add(Dense(384 * 8 * 8, input_dim=latent_size+label_dim, activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    # cnn.add(Dense(384 * 8 * 8, input_dim=latent_size,
    #               kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    # cnn.add(LeakyReLU())
    cnn.add(Reshape((384, 8, 8)))
    cnn.add(LeakyReLU())

    cnn.add(Conv2DTranspose(192, kernel_size=5, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    # cnn.add(Conv2DTranspose(192, kernel_size=5, strides=2, padding='same',
    #                         kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU())

    cnn.add(Conv2DTranspose(96, kernel_size=5, strides=2, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    # cnn.add(Conv2DTranspose(96, kernel_size=5, strides=2, padding='same',
    #                         kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU())

    cnn.add(Conv2DTranspose(48, kernel_size=5, strides=1, padding='same', activation='relu',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU())

    cnn.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh',
                            kernel_initializer='glorot_normal', bias_initializer='Zeros'))

    # this is the z space commonly refered to in GAN papers
    # latent = Input(shape=(latent_size,))

    # this will be our label
    # image_class = Input(shape=(1,), dtype='int32')
    # image_class = Input(shape=(label_dim,), dtype='int32')

    # 10 classes in CIFAR-10
    # cls = Flatten()(Embedding(2, latent_size,
    #                           embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    # h = layers.multiply([latent, cls])

    # YG input
    # inputs = (Concatenate(name='input_concatenation', axis=-1))([latent, image_class])

    latent = (Input(shape=(latent_size,), name='generator_input'))
    label = (Input(shape=(label_dim,), name='generator_label'))
    inputs = (Concatenate(name='input_concatenation'))([latent, label])

    # Original
    # fake_image = cnn(h)

    fake_image = cnn(inputs) #YG changed

    return Model(input=[latent, label], output=fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    # cnn.add(GaussianNoise(0.05, input_shape=(3, 32, 32)))  # Add this layer to prevent D from overfitting!
    cnn.add(GaussianNoise(0.05, input_shape=(3, 64, 64)))  # Add this layer to prevent D from overfitting!

    cnn.add(Conv2D(16, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(32, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(64, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(128, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(256, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(512, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Conv2D(1024, kernel_size=3, strides=2, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))
    #
    cnn.add(Conv2D(2048, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='glorot_normal', bias_initializer='Zeros'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.2))
    cnn.add(Dropout(0.5))

    cnn.add(Flatten())

    cnn.add(MinibatchDiscrimination(50, 30))

    image = Input(shape=(3, 64, 64))
    # image = Input(shape=(3, 32, 32))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation',
                 kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)
    aux = Dense(class_num, activation='softmax', name='auxiliary',
                kernel_initializer='glorot_normal', bias_initializer='Zeros')(features)

    return Model(image, [fake, aux])

def build_generator_model(self):
    kernel_init = 'glorot_uniform'
    model = Sequential(name = 'generator_model')
    model.add(Reshape((1, 1, -1), input_shape=(latent_size+16,)))
    model.add( Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init, ))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Activation('tanh'))
    # 3 inputs
    latent = Input(shape=(self.latent_size, ))
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')
    # embedding
    hairs = Flatten()(Embedding(self.num_class_hairs, 8,  init='glorot_normal')(hairs_class))
    eyes = Flatten()(Embedding(self.num_class_eyes, 8,  init='glorot_normal')(eyes_class))
    # concat_style = merge([hairs, eyes], name='concat_style', mode='concat')
    h = merge([latent, hairs, eyes], mode='concat')
    fake_image = model(h)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    # m.summary()
    return m

def build_discriminator_model(self, num_class = 12):
    kernel_init = 'glorot_uniform'
    discriminator_model = Sequential(name="discriminator_model")
    discriminator_model.add( Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, input_shape=self.image_shape))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Flatten())
    dis_input = Input(shape = self.image_shape)
    features = discriminator_model(dis_input)
    validity = Dense(1, activation="sigmoid")(features)
    label_hair = Dense(self.num_class_hairs, activation="softmax")(features)
    label_eyes = Dense(self.num_class_eyes, activation="softmax")(features)
    m = Model(dis_input, [validity, label_hair, label_eyes])
    # m.summary()
    return m

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 1000
    batch_size = 16
    latent_size = 110

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    data_file_name = 'GAN_RSVP_NORM_GANINPUT.mat'
    # DATA_FOLDER_PATH = '/media/zijing/YG_Storage/ARL_BCIT/Processed EEG Data/X2 RSVP Expertise/YGfixed/Chair/FV/'

    DATA_FOLDER_PATH = '/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/Correct_Data/EEGNET/Input_for_GAN/'

    FILE_PATH = DATA_FOLDER_PATH + '/' + data_file_name

    # mat_2 = scipy.io.loadmat(FILE_PATH)
    mat_2 = h5py.File(FILE_PATH)
    mat_2.keys()

    FeatureV = mat_2['feature']
    labels = mat_2['train_y']

    input_height = 64
    input_width = 64
    output_height = 64
    output_width = 64
    GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

    batch_images = np.zeros((880, 64, 64, 3), dtype=np.float32)

    for idx in range(0, 880):
        # batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_files = ['/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/goodimagetrain/' + str(idx + 1) + '.jpg']
        batch = [
            get_image(batch_file,
                      input_height=input_height,
                      input_width=input_width,
                      resize_height=output_height,
                      resize_width=output_width,
                      ) for batch_file in batch_files]
        batch_images[idx] = np.array(batch).astype(np.float32)

    # X_train = np.reshape(batch_images, [880, 3, 64, 64])

    # build the discriminator, Choose Adam as optimizer according to GANHACK
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size,)) #this makes dim 110
    # image_class = Input(shape=(label_dim,), dtype='int32') #label_dim = 2 for one hot encoding.
    image_class = Input(shape=(label_dim,))

    # get a fake image
    fake = generator([latent, image_class])
    # fake = generator(combined_data)

    # we only want to be able to train generator for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    # nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # X_train = np.reshape(batch_images, [880, 3, 64, 64])
    X_train = np.transpose(batch_images, (0, 3, 1, 2))
    # X_train = np.reshape(batch_images, [880, 3, 64, 64])

    y_train = labels[:, 0:label_dim]

    X_test = X_train[0:100, :, :, :]
    y_test = y_train[0:100, 0:label_dim]
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    if load_weight:
        generator.load_weights('/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/ACGAN/model_save/params_generator_epoch_{0:03d}.hdf5'.format(load_epoch))
        discriminator.load_weights('/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/ACGAN/model_save/params_discriminator_epoch_{0:03d}.hdf5'.format(load_epoch))


    else:
        load_epoch = 0

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(load_epoch + 1, nb_epochs))
        load_epoch += 1
        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.normal(0, 0.5, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            # YG change
            # label_batch = y_train[index * batch_size:(index + 1) * batch_size, :]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, class_num, size= (batch_size, 1))
            one_hot_labels = to_categorical(sampled_labels, num_classes=2)


            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence

            a = sampled_labels.reshape((-1, 1))
            aa = [noise, sampled_labels.reshape((-1, 1))]
            aaa = [noise, one_hot_labels]
            # generated_images = generator.predict(
            #     [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            generated_images = generator.predict(
                aaa, verbose=0)

            disc_real_weight = [np.ones(batch_size), 2 * np.ones(batch_size)]
            disc_fake_weight = [np.ones(batch_size), np.zeros(batch_size)]

            # According to GANHACK, We training our ACGAN-CIFAR10 in Real->D, Fake->D,
            # Noise->G, rather than traditional method: [Real, Fake]->D, Noise->G, actully,
            # it really make sense!

            for train_ix in range(3):
                if index % 30 != 0:
                    X_real = image_batch
                    # Label Soomthing
                    y_real = np.random.uniform(0.7, 1.2, size=(batch_size,))
                    aux_y1 = label_batch.reshape(-1, )
                    epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
                    # Label Soomthing
                    X_fake = generated_images
                    y_fake = np.random.uniform(0.0, 0.3, size=(batch_size,))
                    aux_y2 = sampled_labels

                    # see if the discriminator can figure itself out...
                    epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))
                else:
                    # make the labels the noisy for the discriminator: occasionally flip the labels
                    # when training the discriminator
                    X_real = image_batch
                    y_real = np.random.uniform(0.0, 0.3, size=(batch_size,))
                    # aux_y1 = label_batch[:, 1].reshape(-1, )
                    # aux_y1 = label_batch[:, 1]
                    aux_y1 = label_batch

                    epoch_disc_loss.append(discriminator.train_on_batch(X_real, [y_real, aux_y1]))
                    # Label Soomthing
                    X_fake = generated_images
                    y_fake = np.random.uniform(0.7, 1.2, size=(batch_size,))
                    # aux_y2 = sampled_labels
                    aux_y2 = one_hot_labels

                    # see if the discriminator can figure itself out...
                    epoch_disc_loss.append(discriminator.train_on_batch(X_fake, [y_fake, aux_y2]))
            # make new noise. we generate Guassian Noise rather than Uniform Noise according to GANHACK
            noise = np.random.normal(0, 0.5, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, class_num, 2 * batch_size)
            sampled_labels = to_categorical(sampled_labels, num_classes=2)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.random.uniform(0.7, 1.2, size=(2 * batch_size,))

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(load_epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.normal(0, 0.5, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, class_num, nb_test)
        sampled_labels = to_categorical(sampled_labels, num_classes=2)
        # generated_images = generator.predict(
        #     [noise, sampled_labels.reshape((-1, 1))], verbose=False)
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test.reshape(-1, ), sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)


        # make new noise
        noise = np.random.normal(0, 0.5, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, class_num, 2 * nb_test)
        sampled_labels = to_categorical(sampled_labels, num_classes=2)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            '/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/ACGAN/model_save/params_generator_epoch_{0:03d}.hdf5'.format(load_epoch), True)
        discriminator.save_weights(
            '/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/ACGAN/model_save/params_discriminator_epoch_{0:03d}.hdf5'.format(load_epoch), True)

        # generate some pictures to display
        noise = np.random.normal(0, 0.5, (16, latent_size))
        # sampled_labels = np.array([
        #     [i] * 4 for i in range(4)
        # ]).reshape(-1, 1)
        y_test = y_train[0:100, 0:1]
        sampled_labels = y_test[0:16, 0:label_dim]
        generated_images = generator.predict([noise, sampled_labels]).transpose(0, 2, 3, 1)
        generated_images = np.asarray((generated_images * 127.5 + 127.5).astype(np.uint8))


        def vis_square(data, padsize=1, padval=0):

            # force the number of filters to be square
            n = int(np.ceil(np.sqrt(data.shape[0])))
            padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
            data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

            # tile the filters into an image
            data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
            data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
            return data


        img = vis_square(generated_images)
        if not os.path.exists(path):
            os.makedirs(path)
        Image.fromarray(img).save(
            '/media/eeglab/YG_Storage/ARL_BCIT/Processed_EEG_Data/X2 RSVP Expertise/' \
                       'YGfixed/Chair/ACGAN/images/plot_epoch_{0:03d}_generated.png'.format(load_epoch))

        pickle.dump({'train': train_history, 'test': test_history},
                        open('acgan-history.pkl', 'wb'))
