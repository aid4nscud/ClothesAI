from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tkinter as tk
from tkinter import filedialog
import joblib

# Define target columns
target_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']

# Load the trained model
model_path = 'model.keras'
model = load_model(model_path)

# Load the encoder
encoder_path = 'encoder.pkl' # Path to the saved encoder
encoder = joblib.load(encoder_path) # Assuming you've saved the encoder using joblib

# Function to make prediction on an image
def predict_image(image_path):
    image = load_img(image_path, target_size=(96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image) # Preprocess the image like during training
    prediction = model.predict(image)
    return prediction

# Create a GUI for file selection
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[('JPEG files', '*.jpg')])

# Make prediction on the selected image
if file_path:
    prediction = predict_image(file_path)
    prediction = prediction[0] # Get the first (and only) prediction

    # Split the prediction into groups according to the target columns
    start_idx = 0
    predicted_dict = {}
    for idx, col in enumerate(target_columns):
        categories_count = len(encoder.categories_[idx])
        group_prediction = prediction[start_idx:start_idx + categories_count]
        top_category_idx = np.argmax(group_prediction)
        top_category = encoder.categories_[idx][top_category_idx]
        predicted_dict[col] = top_category
        start_idx += categories_count

    print(predicted_dict)
else:
    print("No file selected.")