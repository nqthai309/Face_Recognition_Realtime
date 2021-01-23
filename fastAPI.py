from fastapi import FastAPI
from pydantic import BaseModel
import utils
import hyper as hp
from numpy import load
from numpy import dot
from numpy.linalg import norm
import cv2


class MyImage(BaseModel):
    bs64_image: str
    name: str = ''


app = FastAPI()


@app.get('/')
async def home():
    return 'home'


@app.post('/predict')
async def predict(image: MyImage):
    # load data
    data = load('feature_data.npz')
    feature_data = data['arr_0']
    label = data['arr_1']
    # preprocessing
    img_RGB = utils.bas64_to_Image(image.bs64_image)
    img_resize = utils.resize_image(img_RGB, hp.REQUIRED_SIZE)
    # embedding
    embedding = utils.get_embedding(hp.model, img_resize)
    # predict
    labelPredict = 'unknow'
    probability = 0.0
    for index in range(len(feature_data)):
        cosine = dot(feature_data[index], embedding) / (norm(feature_data[index]) * norm(embedding))
        # print('cosine = ', cosine)
        if cosine > hp.THRESHOLD:
            labelPredict = label[index]
            probability = cosine
    return probability * 100, labelPredict


@app.post('/getface')
async def getface(image: MyImage):
    img_RGB = utils.bas64_to_Image(image.bs64_image)
    img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
    result = utils.getface(img_BGR, image.name)
    return result


@app.get('/train')
async def train():
    return utils.train()
