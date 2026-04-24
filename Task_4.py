import os
os.environ["KERAS_BACKEND"] = "torch"
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
data_path = r"C:\Users\panch\Downloads\archive (6)\leapGestRecog"
IMG_SIZE = 64

X = []
y = []

gesture_map = {
    '01_palm': 0,
    '02_l': 1,
    '03_fist': 2,
    '04_fist_moved': 3,
    '05_thumb': 4,
    '06_index': 5,
    '07_ok': 6,
    '08_palm_moved': 7,
    '09_c': 8,
    '10_down': 9
}

print("Loading images...")

for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):
        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)

            if gesture in gesture_map:
                label = gesture_map[gesture]

                for img_name in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img_name)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X.append(img)
                        y.append(label)

print("Dataset Loaded Successfully!")
X = np.array(X)
y = np.array(y)

X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu',
                 input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
model.save("hand_gesture_model.keras")
print("Model Saved Successfully!")