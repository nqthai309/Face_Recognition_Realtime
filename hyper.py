from keras.models import load_model

model = load_model('model/facenet_keras.h5')
REQUIRED_SIZE = (160, 160)
THRESHOLD = 0.80
path_train_haar = '/home/thai/PycharmProjects/Face_Recognition_Realtime/database/Haarcascade/'
