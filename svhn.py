import urllib.request
import tarfile
import os
import cv2
import numpy as np
import tensorflow as tf
import Model
import create_data_files
from sklearn.metrics import f1_score
from Model import load_data


def load_model():
    model = tf.keras.models.load_model(
        './SavedModel/svhn_keras_trained_model.h5')
    return model


def down_load_datasets():
    # check if files already exist
    if not os.path.isdir('./Data/train'):
        print("Downloading Training Set")
        train = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
        train_tmp = urllib.request.urlretrieve(train, filename=None)[0]
        print("Extracting Training set")
        tar = tarfile.open(train_tmp)
        tar.extractall('./Data')
    if not os.path.isdir('./Data/test'):
        print("Downloading Test Set")
        test = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'
        test_tmp = urllib.request.urlretrieve(test, filename=None)[0]
        print("Extracting Test Set")
        tar = tarfile.open(test_tmp)
        tar.extractall('./Data')


def convert_image(image, normalise=False):
    image = cv2.imread(image)
    image = cv2.resize(image, dsize=(64, 64))
    image = np.expand_dims(image, 0)
    image = image/255.0
    return image


def interpret_prediction(prediction):
    """Returns interpretable prediction
    """
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    digit_1 = labels[np.argmax(prediction[0])]
    digit_2 = labels[np.argmax(prediction[1])]
    digit_3 = labels[np.argmax(prediction[2])]
    digit_4 = labels[np.argmax(prediction[3])]
    digit_5 = labels[np.argmax(prediction[4])]
    digits = [digit_1, digit_2, digit_3, digit_4, digit_5]
    digits = [x for x in digits if x != 10]
    return digits


def score_model(scores, y_test):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    predicted_digits = np.zeros((len(scores[0][:]), 5))
    true_digits = np.zeros((len(scores[0][:]), 5))
    f1_scores = [None]*5
    full_f1_score = 0
    for x in range(len(scores[0][:])-1):
        for y in range(5):
            predicted_digits[x][y] = labels[np.argmax(scores[y][x])]
            true_digits[x][y] = labels[np.argmax(y_test[x][y])]
    for i in range(5):
        f1_scores[i] = f1_score(
            true_digits[:, i], predicted_digits[:, i], average='macro')
    full_f1_score = np.mean(f1_scores)
    return f1_scores, full_f1_score


def test(image):
    """Returns a string representing the prediction of the model on the
    """
    model = load_model()
    img = convert_image(image)
    prediction = model.predict(img)
    intr_pred = interpret_prediction(prediction)
    intr_pred = "".join(str(x) for x in intr_pred)
    print("The models prediction for the house number is:", intr_pred)


def traintest():
    print("Downloading Datasets")
    down_load_datasets()
    print("Processing Datasets")
    create_data_files.main()
    scores, y_test = Model.main()
    f1_scores, final_f1_score = score_model(scores, y_test)
    print("Average F1 score across each class:", final_f1_score)
    print("F1 score for digit 1:", f1_scores[0])
    print("F1 score for digit 2:", f1_scores[1])
    print("F1 score for digit 3:", f1_scores[2])
    print("F1 score for digit 4:", f1_scores[3])
    print("F1 score for digit 5:", f1_scores[4])
    return final_f1_score

test('1.png')
