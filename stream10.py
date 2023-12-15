# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:51:18 2023

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:30:47 2023

@author: DELL
"""
import streamlit as st
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from ultralytics import YOLO
import pandas as pd
import gdown

# Function to download the model file
def download_file(url, filename):
    gdown.download(url, filename, quiet=False)

# Download the model file
download_file('https://drive.google.com/uc?id=1J753l-T63J5oV-9rK6oJiO_F0RWXXZQk', 'best.pt')  


# Load the model
model = YOLO('best.pt')

# Define the transformation
transform = T.Compose([T.Resize(256),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def predict(image):
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # Process the outputs
    threshold = 0.5
    outputs = [output for output in outputs if output[4] > threshold]

    # Draw bounding boxes on the image and get labels
    labels = []
    for output in outputs:
        # Get the coordinates
        x1, y1, x2, y2 = output[:4]
        # Scale the coordinates to the size of the image
        x1, y1, x2, y2 = x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height
        # Draw the bounding box
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline ="red")
        # Get the label
        labels.append(output[-1])

    # Count the number of objects detected
    counts = len(outputs)

    return counts, labels, image


# Streamlit code to create the interface
st.title("Steel Pipe Detector")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    counts, labels, image = predict(image)
    st.image(image, caption='Detected Image.', use_column_width=True)
    st.write(f"Detected {counts} steel pipes.")
    df = pd.DataFrame({'Label': labels})
    st.table(df)

