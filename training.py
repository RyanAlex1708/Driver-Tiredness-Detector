import os
import argparse
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-e", "--epochs", type=int, default=25,
                help="number of epochs to train for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output training plot")
args = vars(ap.parse_args())


data = []
labels = []
imagePaths = [os.path.join(args["dataset"], f) for f in os.listdir(args["dataset"])]
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] skipping invalid image: {imagePath}")
        continue 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]  
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")  
])
opt = Adam(learning_rate=1e-3)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32, epochs=args["epochs"], verbose=1)

print("[INFO] saving model...")
model.save("eye_detector.model", save_format="h5")
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
print(f"[INFO] training plot saved to {args['plot']}")