import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "training data")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # os.path.dirname(path) below could also be root, but this is more readible.
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() 
            #print(label, path)

            # filling the label_ids dictionary with key and values (key being label, value being the id )
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label] #getting current id to later append to y_labels
            
            #y_labels.append(label) # some number
            #x_train.append(path) # convert to NUMPY array, GRAY
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            #! look into how Numpy does this stuff
            #! more specifically, what are the values in the numpy array
            image_array = np.array(pil_image, "uint8")
            #print(image_array)

            # Constructing region of interest and then adding them to training data
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml") 