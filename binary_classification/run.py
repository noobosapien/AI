import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array

data_path = "../datasets/heart.csv"

heart_df = pd.read_csv(data_path)

# for i in heart_df.columns:
#     sns.boxplot(x=heart_df[i])
#     plt.show()

df = heart_df.copy()
# df.info()
# df.target = df.target.map({0: 'Healthy', 1: 'Heart Patient'})
# sns.countplot(x='target', data=df)

# df.sex = df.sex.map({0: 'Healthy', 1: 'Heart Patient'})
# sns.countplot(x='sex', data=df, hue=df.target)
# plt.show()

# plt.hist(df[df.target=='Heart Patient']['age'],color='b', alpha=0.5, bins=15, label='Patient')
# plt.hist(df[df.target=='Healthy']['age'],color='g', alpha=0.5, bins=15, label='Healthy Patient')
# plt.legend()
# plt.show()

# plt.hist(df[df.target=='Heart Patient']['thalach'],color='b', alpha=0.5, bins=15, label='Patient')
# plt.hist(df[df.target=='Healthy']['thalach'],color='g', alpha=0.5, bins=15, label='Healthy Patient')
# plt.legend()
# plt.show()

x = df.drop('target', axis=1)
y = df['target']

model = keras.Sequential()
model.add(layers.Dense(11, activation='relu', input_dim = 13))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x, y, validation_split=0.22, epochs = 600)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

xnew = array([[67, 1, 0, 125, 254, 1, 1, 163, 0, 0.2,1,2,3]])
xnew = array(xnew, dtype=np.float64)

ynew = (model.predict(xnew) > 0.5).astype('int32')
print(ynew[0])