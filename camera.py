import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model.weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    # Use webcam (0 is usually the default webcam)
    video = cv2.VideoCapture('C:/Users/kmoha/Downloads/Accident-Detection-System-st josephs/accidentvideo.mp4')
    if not video.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Can't receive frame from webcam. Exiting ...")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
            
        cv2.imshow('Video', frame)
    
    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()