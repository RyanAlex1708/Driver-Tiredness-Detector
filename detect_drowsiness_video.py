from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from send_serial_testing import flash_light

counter = 10

# ------------------ DETECTION FUNCTION ------------------
def detect_and_predict(frame, faceNet, eyeNet, yawnNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]

            if face.size > 0:
                # -------- EYE MODEL --------
                face_eye = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_eye = cv2.resize(face_eye, (224, 224))
                face_eye = img_to_array(face_eye)
                face_eye = preprocess_input(face_eye)
                face_eye = np.expand_dims(face_eye, axis=0)

                eye_pred = eyeNet.predict(face_eye)[0]

                # -------- YAWN MODEL --------
                face_yawn = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_yawn = cv2.resize(face_yawn, (224, 224))
                face_yawn = img_to_array(face_yawn)
                face_yawn = face_yawn / 255.0
                face_yawn = np.expand_dims(face_yawn, axis=0)

                yawn_pred = yawnNet.predict(face_yawn, verbose=0)
                yawn = float(np.squeeze(yawn_pred))

                results.append((startX, startY, endX, endY, eye_pred, yawn))

    return results



ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector")
ap.add_argument("-m", "--model", type=str, default="eye_detector.model")
args = vars(ap.parse_args())


print("[INFO] loading face detector...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading eye model...")
eyeNet = load_model("eye_detector.model", compile=False)

print("[INFO] loading yawn model...")
yawnNet = load_model("yawn_detection_model.h5", compile=False)


print("[INFO] starting video stream...")
vs = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(2.0)


while True:
    ret, frame = vs.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=400)

    results = detect_and_predict(frame, faceNet, eyeNet, yawnNet)

    print("Faces detected:", len(results))  # DEBUG

    for (startX, startY, endX, endY, eye_pred, yawn) in results:
        (eye, withoutEye) = eye_pred


        if eye > withoutEye:
            label = "Awake"
            color = (0, 255, 0)
        else:
            label = "Sleepy"
            color = (0, 0, 255)


        if yawn > 0.5:
            yawn_label = "Yawning"
        else:
            yawn_label = "No Yawn"
        print("Yawn Value:", yawn)


        if label == "Sleepy":
            counter -= 1
            if counter <= 0:
                flash_light(label)
                time.sleep(2)
                counter = 10
        else:
            counter = 10


        label_text = "{}: {:.2f}%".format(label, max(eye, withoutEye) * 100)
        yawn_text = "Yawn: {:.2f}%".format(yawn * 100)


        cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.putText(frame, yawn_text, (startX, startY - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
vs.release()