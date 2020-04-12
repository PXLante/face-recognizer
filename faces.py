import numpy as np
import cv2
import pickle

# Play around with these classifiers and find one that works best for what we're doing
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray is turned into a NUMPY array here
    # Look at documentation
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
           # region of interest
           print(x, y, w, h)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = frame[y:y+h, x:x+w]

           # recognize? deep learned model predict keras tensorflow pytorch scikit learn
           id_, conf = recognizer.predict(roi_gray) # confidence is a weird value
           if (conf>= 85):
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)  

           img_item = "my-image-color.png"
           cv2.imwrite(img_item, roi_color)  

           # Drawing the box around the face
           color = (255, 0, 0) #BGR 0-255
           stroke = 2
           end_coord_x = x + w # width
           end_coord_y = y + h # height
           cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()