import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import keras.backend as K
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import L1L2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets, fix_names
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_unfix, dim_ordering_shape


# This line allows mpl to run with no DISPLAY defined
mpl.use("Agg")


def model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Conv2D(int(nch / 2), (h, h), padding="same", kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 2), (h, h), padding="same", kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 4), (h, h), padding="same", kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (h, h), padding="same", kernel_regularizer=reg()))
    model.add(Activation("sigmoid"))
    return model


def model_discriminator():
    nch = 256
    h = 5
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)

    c1 = Conv2D(int(nch / 4),
                (h, h),
                padding="same",
                kernel_regularizer=reg(),
                input_shape=dim_ordering_shape((3, 32, 32)))
    c2 = Conv2D(int(nch / 2),
                (h, h),
                padding="same",
                kernel_regularizer=reg())
    c3 = Conv2D(nch,
                (h, h),
                padding="same",
                kernel_regularizer=reg())
    c4 = Conv2D(1,
                (h, h),
                padding="same",
                kernel_regularizer=reg())

    def m(dropout):
        model = Sequential()
        model.add(c1)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c2)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c3)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c4)
        model.add(AveragePooling2D(pool_size=(4, 4), padding="valid"))
        model.add(Flatten())
        model.add(Activation("sigmoid"))
        return model
    return m


def cifar10_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)


def ioswp_data():
    from keras.preprocessing.image import ImageDataGenerator
    from skimage.io import imread
    
    train_dir = r'D:\Pictures\iOS_Wallpaper\data\dataset\train'
    test_dir = r'D:\Pictures\iOS_Wallpaper\data\dataset\test'

    train_dir = r'D:\Pictures\iOS_Wallpaper\data\hah\train'
    test_dir = r'D:\Pictures\iOS_Wallpaper\data\hah\test'
    batch_size = 13

    xtrain = np.array([
        imread(os.path.join(train_dir, filename))
        for filename in os.listdir(train_dir)
    ])
    xtest = np.array([
        imread(os.path.join(test_dir, filename))
        for filename in os.listdir(test_dir)
    ])

    xtrain, xtest = cifar10_process(xtrain), cifar10_process(xtest)

    train_datagen = ImageDataGenerator(
        rotation_range=90, # 90°まで回転
        width_shift_range=0.1, # 水平方向にランダムでシフト
        height_shift_range=0.1, # 垂直方向にランダムでシフト
        # channel_shift_range=50.0, # 色調をランダム変更
        # shear_range=0.39, # 斜め方向(pi/8まで)に引っ張る
        horizontal_flip=True, # 垂直方向にランダムで反転
        vertical_flip=True # 水平方向にランダムで反転
    )
    train_datagen.fit(xtrain)

    test_datagen = ImageDataGenerator(
        rotation_range=90, # 90°まで回転
        width_shift_range=0.1, # 水平方向にランダムでシフト
        height_shift_range=0.1, # 垂直方向にランダムでシフト
        # channel_shift_range=50.0, # 色調をランダム変更
        # shear_range=0.39, # 斜め方向(pi/8まで)に引っ張る
        horizontal_flip=True, # 垂直方向にランダムで反転
        vertical_flip=True # 水平方向にランダムで反転
    )
    test_datagen.fit(xtest)

    def train_generator():
        g = train_datagen.flow(
            xtrain,
            batch_size=batch_size
        )
        for d in g:
            yield (d, gan_targets(batch_size), None)

    def test_generator():
        g = test_datagen.flow(
            xtest,
            batch_size=batch_size
        )
        for d in g:
            yield (d, gan_targets(batch_size), None)

    return train_generator(), test_generator(), xtrain, xtest


def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch,
                generator, discriminator, latent_dim,
                targets=gan_targets, loss="binary_crossentropy"):
    csvpath = os.path.join(path, "history.csv")
    if os.path.exists(csvpath):
        print("Already exists: {}".format(csvpath))
        return

    print("Training: {}".format(csvpath))
    # gan (x - > yfake, yreal), z is gaussian generated on GPU
    # can also experiment with uniform_latent_sampling
    d_g = discriminator(0)
    d_d = discriminator(0.5)
    generator.summary()
    d_d.summary()
    gan_g = simple_gan(generator, d_g, None)
    gan_d = simple_gan(generator, d_d, None)
    x = gan_g.inputs[1]
    z = normal_latent_sampling((latent_dim,))(x)
    # estiminate z from inputs
    gan_g = Model([x], fix_names(gan_g([z, x]), gan_g.output_names))
    gan_d = Model([x], fix_names(gan_d([z, x]), gan_d.output_names))

    # build adversarial model
    model = AdversarialModel(player_models=[gan_g, gan_d],
                             player_params=[generator.trainable_weights,
                                            d_d.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    # create callback to generate images
    zsamples = np.random.normal(size=(10 * 10, latent_dim))

    def generator_sampler():
        xpred = generator.predict(zsamples)
        xpred = dim_ordering_unfix(xpred.transpose((0, 2, 3, 1)))
        return xpred.reshape((10, 10) + xpred.shape[1:])

    generator_cb = ImageGridCallback(
                    os.path.join(path, "epoch-{:03d}.png"),
                    generator_sampler, cmap=None)

    callbacks = [generator_cb]
    if K.backend() == "tensorflow":
        callbacks.append(
            TensorBoard(log_dir=os.path.join(path, "logs"),
                        histogram_freq=0, write_graph=True, write_images=True))

    # train model
    """
    xtrain, xtest = cifar10_data()
    model.add(UpSampling2D(size=(2, 2)))
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest),
                        callbacks=callbacks, epochs=nb_epoch,
                        batch_size=32)
    """

    # train_generator, validation_generator = ioswp_data()
    train_generator, validation_generator, xtrain, xtest = ioswp_data()
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    history = model.fit_generator(
        train_generator,
        # validation_data=(xtest, ytest),
        validation_data=validation_generator,
        callbacks=callbacks,
        epochs=nb_epoch,
        steps_per_epoch=(143 / 13) * 1000,
        validation_steps=(143 / 13) * 100
    )

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)

    # save models
    generator.save(os.path.join(path, "generator.h5"))
    d_d.save(os.path.join(path, "discriminator.h5"))


def main():
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    # generator (z -> x)
    generator = model_generator()
    # discriminator (x -> y)
    discriminator = model_discriminator()
    if not os.path.exists("output/gan-ioswp"):
        os.mkdir("output/gan-ioswp")
    example_gan(AdversarialOptimizerSimultaneous(), "output/gan-ioswp",
                opt_g=Adam(1e-4, decay=1e-5),
                opt_d=Adam(1e-3, decay=1e-5),
                nb_epoch=100, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":
    main()
