import os
import h5py
from create_data_files_folders import ImageProcessor, ImageProcessor
import tensorflow as tf
from pathlib import Path
import numpy as np
import pickle
import joblib
from itertools import tee
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
data_dir = './Data/processed'


def load_data(dataset='cropped', normalised=True, grey=False):
    """"Returns the dataset chosen by parameters, only use one of normalised/grey/plain"

    Keyword Arguments:
        dataset {string} -- [Name of dataset to load] (default: {cropped})
        normalised {bool} -- [Normalised dataset] (default: {True})
        grey {bool} -- [greyscale dataset] (default: {False})
    """
    if normalised:
        directory = 'normalised'
    elif grey:
        directory = 'grey'

    image_processor = ImageProcessor(data_dir)
    train_data, train_labels = image_processor.load_data('train', dataset)
    test_data, test_labels = image_processor.load_data('test', dataset)

    return train_data, train_labels, test_data, test_labels


def create_model(dim, channel):
    input_layer = tf.keras.Input(shape=(dim, dim, channel))
    cl1 = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5),
                                 padding='same', input_shape=(dim, dim, channel),
                                 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(input_layer)
    bnl1 = tf.keras.layers.BatchNormalization(axis=-1)(cl1)
    mpl1 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl1)
    dpl1 = tf.keras.layers.Dropout(rate=0.4)(mpl1)

    # second layer
    cl2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                 padding='same', activation='relu', use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001))(dpl1)
    bnl2 = tf.keras.layers.BatchNormalization(axis=-1)(cl2)
    mpl2 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl2)
    dpl2 = tf.keras.layers.Dropout(rate=0.4)(mpl2)

    # third layer
    cl3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), use_bias=True,
                                 padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl2)
    bnl3 = tf.keras.layers.BatchNormalization(axis=-1)(cl3)
    mpl3 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl3)
    dpl3 = tf.keras.layers.Dropout(rate=0.4)(mpl3)

    # Fourth layer
    cl4 = tf.keras.layers.Conv2D(filters=160, kernel_size=(5, 5),
                                 padding='same', activation='relu', use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl3)
    bnl4 = tf.keras.layers.BatchNormalization(axis=-1)(cl4)
    mpl4 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl4)
    dpl4 = tf.keras.layers.Dropout(rate=0.4)(mpl4)

    # fifth layer
    cl5 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu', use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl4)
    bnl5 = tf.keras.layers.BatchNormalization(axis=-1)(cl5)
    mpl5 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl5)
    dpl5 = tf.keras.layers.Dropout(rate=0.4)(mpl5)

    # sixth layer
    cl6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu', use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl5)
    bnl6 = tf.keras.layers.BatchNormalization(axis=-1)(cl6)
    mpl6 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl6)
    dpl6 = tf.keras.layers.Dropout(rate=0.4)(mpl6)
    # seventh layer
    cl7 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl6)
    bnl7 = tf.keras.layers.BatchNormalization(axis=-1)(cl7)
    mpl7 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl7)
    dpl7 = tf.keras.layers.Dropout(rate=0.4)(mpl7)

    # eigth layer
    cl8 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001),
                                 bias_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl7)
    bnl8 = tf.keras.layers.BatchNormalization(axis=-1)(cl8)
    mpl8 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl8)
    dpl8 = tf.keras.layers.Dropout(rate=0.4)(mpl8)

    # 1st fully connected layer
    fl1 = tf.keras.layers.Flatten()(dpl8)
    dl1 = tf.keras.layers.Dense(3072, activation='relu', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                bias_regularizer=tf.keras.regularizers.l2(
                                    0.001))(fl1)
    dpl9 = tf.keras.layers.Dropout(rate=0.4)(dl1)
    # 2nd fully connected layer
    dl2 = tf.keras.layers.Dense(3072, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                bias_regularizer=tf.keras.regularizers.l2(
                                    0.001))(dpl9)
    dpl10 = tf.keras.layers.Dropout(rate=0.4)(dl2)

    # output layer
    output_length = tf.keras.layers.Dense(6, activation='softmax')(dpl10)
    output_digit1 = tf.keras.layers.Dense(11, activation='softmax')(dpl10)
    output_digit2 = tf.keras.layers.Dense(11, activation='softmax')(dpl10)
    output_digit3 = tf.keras.layers.Dense(11, activation='softmax')(dpl10)
    output_digit4 = tf.keras.layers.Dense(11, activation='softmax')(dpl10)
    output_digit5 = tf.keras.layers.Dense(11, activation='softmax')(dpl10)

    # create model
    model = tf.keras.models.Model(
        input_layer, [output_length, output_digit1, output_digit2, output_digit3, output_digit4, output_digit5])

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001, 10000, 0.95)
    adam = tf.keras.optimizers.Adam(lr=0.00001)
    rms = tf.keras.optimizers.RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr_schedule), metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='svhnCNNModel.png', show_shapes=True)
    return model


def train_model(model, generator, cropped=True):
    batch_size = 250
    num_classes = 11
    epochs = 200
    save_dir = os.path.join(
        os.getcwd(), 'Saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = 'svhn_keras_trained_model_low_drop.h5'
    model_path = os.path.join(save_dir, model_name)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0.001, patience=10,
                                                     restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_dir+'/best_model_low_drop.h5', monitor='val_loss', save_best_only=True, mode='min')
    train_df = joblib.load('./Data/processed/train/metadata_df.h5')
    extra_df = joblib.load('./Data/processed/extra/metadata_df.h5')
    df = train_df.append(extra_df, ignore_index=True)
    if cropped is True:
        train_gen = generator.flow_from_dataframe(df,
                                                  target_size=(54, 54), batch_size=batch_size,
                                                  x_col='filename', y_col='label', class_mode='raw', subset='training',)
        train_samples = train_gen.samples
        train_gen = generator_wrapper(train_gen)
        val_gen = generator.flow_from_dataframe(df,
                                                target_size=(54, 54), batch_size=250, x_col='filename', y_col='label', class_mode='raw', subset='validation')
        val_samples = val_gen.samples
        val_gen = generator_wrapper(val_gen)
    history = model.fit_generator(train_gen, steps_per_epoch=train_samples//batch_size, callbacks=[
                                  early_stopper, checkpoint], epochs=epochs, validation_data=val_gen, validation_steps=val_samples//batch_size, workers=-1, use_multiprocessing=True)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    return model


def predict_model(model, generator, cropped=True):
    df = joblib.load('./Data/processed/test/metadata_df.h5')
    test_gen = generator.flow_from_dataframe(df, target_size=(
        54, 54), batch_size=250, x_col='filename', y_col='label', class_mode='raw', shuffle=False)
    test_samples = test_gen.samples
    y_true = get_labels(test_gen)
    test_gen = generator_wrapper(test_gen)
    predictions = model.predict_generator(test_gen,
                                          steps=test_samples//250, workers=-1, use_multiprocessing=True)
    return predictions, y_true, test_samples


def get_labels(generator):
    im_processor = ImageProcessor(data_dir)
    labels = []
    j = 0
    for batch_x, batch_y in generator:
        lengths = np.zeros((len(batch_y), 1))
        y = np.zeros((len(batch_y), 5, 11))
        for i in range(0, len(batch_y)):
            lengths[i] = len(batch_y[i])
            y[i] = im_processor.create_label_array(
                [x for x in batch_y[i]])
            j += 1
        y_label1 = y[:, 0, :]
        y_label2 = y[:, 1, :]
        y_label3 = y[:, 2, :]
        y_label4 = y[:, 3, :]
        y_label5 = y[:, 4, :]
        y_labels = [lengths, y_label1, y_label2, y_label3, y_label4, y_label5]
        lengths = tf.keras.utils.to_categorical(lengths, num_classes=6)
        labels.append(y_labels)
        print('Completed: {}'.format(j*100/generator.samples))

        if j == generator.samples:
            return labels


def generator_wrapper(generator):
    im_processor = ImageProcessor(data_dir)
    for batch_x, batch_y in generator:
        lengths = np.zeros((len(batch_y), 1))
        y = np.zeros((len(batch_y), 5, 11))
        for i in range(0, len(batch_y)):
            lengths[i] = len(batch_y[i])
            y[i] = im_processor.create_label_array(
                [x for x in batch_y[i]])
        y_label1 = y[:, 0, :]
        y_label2 = y[:, 1, :]
        y_label3 = y[:, 2, :]
        y_label4 = y[:, 3, :]
        y_label5 = y[:, 4, :]
        lengths = tf.keras.utils.to_categorical(lengths, num_classes=6)
        y_labels = [lengths, y_label1, y_label2, y_label3, y_label4, y_label5]
        yield (batch_x, y_labels)


def main():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    dim = 54
    channel = 3
    model = create_model(dim, channel)
    model = train_model(model, train_datagen, cropped=True)
    #model = tf.keras.models.load_model('./SavedModel/best_model_low_drop.h5')
    predictions, y_true, num_samples = predict_model(model, test_datagen)
    return predictions, y_true, num_samples


if __name__ == "__main__":
    main()
