from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("adult_image_classifier.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    #print(pred)  # optional: remove if you don't want to see raw output
    return "Adult" if pred[0][0] < 0.5 else "Non-Adult"

def predict_all_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            prediction = predict_image(img_path)
            print(f"{filename}: {prediction}")

# Example usage
predict_all_images('/home/sanket/Pictures')
#predict_all_images('test_images')
