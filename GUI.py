from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from threading import Thread
import requests
import base64
from io import BytesIO
from PIL import Image

model_path = 'model/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(model_path)

window = Tk()
window.title('Face recognition')
window.geometry('640x560')
video = cv2.VideoCapture(0)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
canvas = Canvas(window, width=canvas_w, height=canvas_h, bg='white')
canvas.pack()

photo = None
count = 0
flag = 0
name_old = ''
Probability = ''


def numpy2base64(numpyarray):
    img = Image.fromarray(numpyarray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    bytes_data = output_buffer.getvalue()
    bs64_data = base64.b64encode(bytes_data)
    return str(bs64_data)


def handleTest(x, y, w, h, frame):
    global name_old, Probability
    bs64_data = numpy2base64(frame[y:y + h, x:x + w])
    response = requests.post('http://127.0.0.1:8000/predict', json={'bs64_image': str(bs64_data)})
    response = response.json()
    class_probability = response[0]
    predict_names = response[1]
    name_old = predict_names
    Probability = str(int(class_probability)) + ' %'


def handleGetFace():
    global flag, count
    if nameTrain.get() == '' or count == 0:
        threadMessage = Thread(target=ShowMessage, args=('cant detect face in frame'
                                                         ' or new name is empty !!',))
        threadMessage.start()
    else:
        flag = 1


def ShowMessage(text):
    messagebox.showwarning(message=text)


buttonTrain = Button(window, text='Get Face', command=handleGetFace)
buttonTrain.place(x=300, y=485)
labelTrain = Label(window, text='New Name:')
labelTrain.place(x=2, y=490)
nameTrain = Entry(window, width=25)
nameTrain.place(x=80, y=490)


def update_frame():
    global canvas, photo, count, flag, name_old, Probability
    count += 1
    _, frame = video.read()
    frame2 = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=7, minSize=(70, 70),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if faces == ():
        # if frame no object
        count = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name_old, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, Probability, (x + 120, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if flag == 1:
            # getface in frame
            count = 0
            newName = nameTrain.get()
            bs64_data = numpy2base64(frame2[y:y + h, x:x + w])
            response = requests.post('http://127.0.0.1:8000/getface', json={'bs64_image': str(bs64_data),
                                                                            'name': str(newName)})
            flag = 0
            nameTrain.delete(0, END)
            threadShowMessageDONE = Thread(target=ShowMessage, args=(str(response.json()),))
            threadShowMessageDONE.start()
        else:
            # predict turn
            if count == 8:
                count = 0
                threadTest = Thread(target=handleTest(x, y, w, h, frame))
                threadTest.start()
            else:
                pass
            break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    window.after(10, update_frame)


update_frame()

window.mainloop()
