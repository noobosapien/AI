import numpy as np
from numpy import array
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_path = "../datasets/kc_house_data.csv"

df = pd.read_csv(data_path)
df['reg_year'] = df['date'].str[:4]
df['reg_year'] = df['reg_year'].astype('int')
df.dtypes
df['house_age'] = np.NaN


for i, j in enumerate(df['yr_renovated']):
    if(j ==0):
        df['house_age'][i] = df['reg_year'][i] - df['yr_built'][i]
    else:
        df['house_age'][i] = df['reg_year'][i] - df['yr_renovated'][i]

df.drop(['date', 'yr_built', 'yr_renovated', 'reg_year',
         'id', 'zipcode', 'lat', 'long'], axis=1, inplace=True)
df = df[df['house_age'] != -1]

# for i in df.columns:
#     sns.displot(df[i])
#     plt.show()

# plt.figure()
# sns.pairplot(df)
# plt.show()

X = df.drop('price', axis=1)
Y = df['price']

model = keras.Sequential()
model.add(layers.Dense(14, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X, Y, validation_split=0.33, batch_size=32, epochs=30)

model.summary()

Xnew = array([[2,3,1280, 5550, 1,0,0,4,7,2280,0,1440,5750,60]])
Xnew = array(Xnew, dtype=np.float64)
Ynew = model.predict(Xnew)
print(Ynew[0])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('no of epochs')
plt.legend(['training', 'test'], loc='upper left')
plt.show()
