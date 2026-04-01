import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("soil_model.h5")

classes = [
    "Alluvial Soil",
    "Arid Soil",
    "Black Soil",
    "Laterite Soil",
    "Mountain Soil",
    "Red Soil",
    "Yellow Soil"
]

img_path = "test.jpg"   # your test image

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

result = classes[np.argmax(prediction)]

print("Predicted Soil Type:", result)