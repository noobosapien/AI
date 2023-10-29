from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

model = load_model('../models/flowers.h5')

image = load_img('./predict_images/2.jpg', target_size=(100,100))

img = img_to_array(image)
img = img.reshape(1,100,100,3)

result = model.predict(img)
class_labels = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
result = np.argmax(result)
print([key for key in class_labels][result])