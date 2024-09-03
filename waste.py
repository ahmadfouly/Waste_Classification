import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load your model
model = tf.keras.models.load_model('resnet_model.h5')

# Define your class names 
class_names = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

def load_image(image_file):
    """Load and resize the image to be suitable for the model."""
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict(image):
    """Run model prediction on the image and return scores."""
    img_array = load_image(image)
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0]) 
    return scores

def main():
    st.title("Waste Classification App")
    uploaded_file = st.file_uploader("Upload an image of waste to classify", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        scores = predict(image)
        result_index = np.argmax(scores)
        result_confidence = np.max(scores)

        st.write(f"This image most likely belongs to {class_names[result_index]} with a {result_confidence * 100:.2f}% confidence.")

        st.write("\nAll Probabilities:")
        for i, score in enumerate(scores):
            st.write(f"Class: {class_names[i]}\t\tProbability: {score * 100:.2f}%")

if __name__ == '__main__':
    main()
