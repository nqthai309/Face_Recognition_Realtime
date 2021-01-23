import numpy as np
import base64
from PIL import Image
from io import BytesIO
import cv2
import os
import hyper as hp
from numpy import savez_compressed


def bas64_to_Image(base64_data):
    base64_data = base64_data[1:]
    im_bytes = base64.b64decode(base64_data)
    im_file = BytesIO(im_bytes)
    image = Image.open(im_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def resize_image(img, required_size):
    img = Image.fromarray(img)
    img = img.resize(required_size)
    img = np.asarray(img)
    return img


def get_embedding(model, img_array):
    face_pixels = img_array.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return np.asarray(yhat[0])


def getface(img_array, newName):
    if not os.path.exists(hp.path_train_haar + newName):
        os.makedirs(hp.path_train_haar + newName)
        cv2.imwrite(hp.path_train_haar + newName + '/image.jpg', img_array)
        return 'finish'
    else:
        text = 'The name already exists in the database, '
        return text


def train():
    X, y = list(), list()
    for subdir in os.listdir(hp.path_train_haar):
        path = hp.path_train_haar + subdir + '/'
        for data in os.listdir(path):
            image = Image.open(path + data)
            image = image.convert('RGB')
            image = image.resize((160, 160))
            pixels = np.asarray(image)
            X.extend(([pixels]))
            y.extend([subdir])
    trainX = np.asarray(X)
    trainy = np.asarray(y)
    newTrainX = list()
    for face_pixel in trainX:
        embedding = get_embedding(hp.model, face_pixel)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    savez_compressed('feature_data.npz', newTrainX, trainy)
    return 'done!!'
