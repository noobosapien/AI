import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array

data_path = "../datasets/winequality-red.csv"

wine_df = pd.read_csv(data_path)

df = wine_df.copy()

# df.quality = df.quality.map({0:'0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10'})
# sns.countplot(x='quality', data=df)
# plt.show()

x = wine_df.drop('quality', axis=1)
y = wine_df['quality']

model = keras.Sequential()
model.add(layers.Dense(11, activation='relu', input_dim=11))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, batch_size=100, epochs=20, validation_split=0.22)
model.summary()
model.save('../models/multiclass.h5')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

xnew = array([[7.9,0.6,0.06,1.6,0.069,15.0,59.0,0.9964,3.3,0.46,9.4]])
xnew = array(xnew, dtype=np.float64)
ynew = model.predict(xnew) 

max_index_row = np.argmax(ynew, axis=1)
print(max_index_row)