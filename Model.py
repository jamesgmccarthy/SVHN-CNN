import os
import h5py
from create_data_files import ImageProcessor
import tensorflow as tf
import numpy as np
import pickle

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
    train_data, train_labels = image_processor.load_data('otrain', dataset)
    test_data, test_labels = image_processor.load_data('test', dataset)

    return train_data, train_labels, test_data, test_labels


def create_model(X, y):
    input_layer = tf.keras.Input(shape=(X.shape[1:]))
    cl1 = tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5),
                                 padding='same', input_shape=X.shape[1:],
                                 activation='relu', use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(input_layer)
    bnl1 = tf.keras.layers.BatchNormalization(axis=-1)(cl1)
    mpl1 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl1)
    dpl1 = tf.keras.layers.Dropout(rate=0.5)(mpl1)

    # second layer
    cl2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl1)
    bnl2 = tf.keras.layers.BatchNormalization(axis=-1)(cl2)
    mpl2 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl2)
    dpl2 = tf.keras.layers.Dropout(rate=0.5)(mpl2)

    # third layer
    cl3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl2)
    bnl3 = tf.keras.layers.BatchNormalization(axis=-1)(cl3)
    mpl3 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl3)
    dpl3 = tf.keras.layers.Dropout(rate=0.5)(mpl3)

    # Fourth layer
    cl4 = tf.keras.layers.Conv2D(filters=160, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl3)
    bnl4 = tf.keras.layers.BatchNormalization(axis=-1)(cl4)
    mpl4 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl4)
    dpl4 = tf.keras.layers.Dropout(rate=0.5)(mpl4)

    # fifth layer
    cl5 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl4)
    bnl5 = tf.keras.layers.BatchNormalization(axis=-1)(cl5)
    mpl5 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl5)
    dpl5 = tf.keras.layers.Dropout(rate=0.5)(mpl5)

    # sixth layer
    cl6 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl5)
    bnl6 = tf.keras.layers.BatchNormalization(axis=-1)(cl6)
    mpl6 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(1, 1), padding='same')(bnl6)
    dpl6 = tf.keras.layers.Dropout(rate=0.5)(mpl6)
    # seventh layer
    cl7 = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5),
                                 padding='same', activation='relu',
                                 use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(
                                     0.001))(dpl6)
    bnl7 = tf.keras.layers.BatchNormalization(axis=-1)(cl7)
    mpl7 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='same')(bnl7)
    dpl7 = tf.keras.layers.Dropout(rate=0.5)(mpl7)
    # 1st fully connected layer
    fl1 = tf.keras.layers.Flatten()(dpl7)
    dl1 = tf.keras.layers.Dense(3072, activation='relu')(fl1)
    dpl8 = tf.keras.layers.Dropout(rate=0.5)(dl1)
    # 2nd fully connected layer
    dl2 = tf.keras.layers.Dense(3072, activation='relu')(dpl8)
    dpl9 = tf.keras.layers.Dropout(rate=0.5)(dl2)

    # output layer
    output_digit1 = tf.keras.layers.Dense(11, activation='softmax')(dpl9)
    output_digit2 = tf.keras.layers.Dense(11, activation='softmax')(dpl9)
    output_digit3 = tf.keras.layers.Dense(11, activation='softmax')(dpl9)
    output_digit4 = tf.keras.layers.Dense(11, activation='softmax')(dpl9)
    output_digit5 = tf.keras.layers.Dense(11, activation='softmax')(dpl9)

    # create model
    model = tf.keras.models.Model(
        input_layer, [output_digit1, output_digit2, output_digit3, output_digit4, output_digit5])

    # optimizer
    adam = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='svhnCNNModel.png', show_shapes=True)
    return model


def train_model(model, X, y):
    X = X/255.0
    batch_size = 250
    num_classes = 11
    epochs = 200
    save_dir = os.path.join(
        os.getcwd(), '.SavedModels/multidigit')
    model_name = 'svhn_keras_trained_model.h5'
    y_label1 = y[:, 0, :]
    y_label2 = y[:, 1, :]
    y_label3 = y[:, 2, :]
    y_label4 = y[:, 3, :]
    y_label5 = y[:, 4, :]
    y_labels = [y_label1, y_label2, y_label3, y_label4, y_label5]
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0.001, patience=30,
                                                     restore_best_weights=True)
    history = model.fit(X, y_labels, batch_size=batch_size,
                        epochs=epochs, validation_split=0.1, callbacks=[early_stopper])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    with open('./Data/saved_models/multidigit_history', 'wb') as f:
        pickle.dump(history.history, f)
    return model


def test_model(model, X, y):
    X = X/255.0
    y_label1 = y[:, 0, :]
    y_label2 = y[:, 1, :]
    y_label3 = y[:, 2, :]
    y_label4 = y[:, 3, :]
    y_label5 = y[:, 4, :]
    y_labels = [y_label1, y_label2, y_label3, y_label4, y_label5]
    scores = model.evaluate(X, y_labels)
    print("Loss of model", scores[0])
    print("Accuracy", scores[1])
    predictions = model.predict(X)
    return predictions, y


def main():
    X_train, y_train, X_test, y_test = load_data(dataset='cropped')
    model = create_model(X_train, y_train)
    model = train_model(model, X_train, y_train)
    scores, y = test_model(model, X_test, y_test)
    return scores, y


if __name__ == "__main__":
    main()
