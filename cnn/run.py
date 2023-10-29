
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.2,2.0],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

model = keras.Sequential()

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation='softmax'))

training_iterator = train_datagen.flow_from_directory('../datasets/flowers/train', batch_size=64, target_size=(100, 100))

testing_iterator = test_datagen.flow_from_directory('../datasets/flowers/test', batch_size=64, target_size=(100,100))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(training_iterator, validation_data=testing_iterator, epochs=8)

model.summary()
model.save('../models/flowers.h5')
#{'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('no of epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()
