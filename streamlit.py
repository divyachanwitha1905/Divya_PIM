# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:30:47 2023

@author: DELL
"""

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import requests
from ultralytics import YOLO

# Function to download the model file
def download_file(url, filename):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Download the model file
download_file('https://drive.google.com/uc?id=18vfcRbw3hkaggwGvforcPVXHA1CYTz-8', 'best.pt')

# Load the model
model = YOLO('best.pt')

# Define the transformation
transform = T.Compose([T.Resize(256),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def predict(image):
    # Transform the image
    image = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)

    # Process the outputs
    threshold = 0.5
    outputs = [output for output in outputs if len(output) > 4 and output[4] > threshold]

    # Count the number of objects detected
    counts = len(outputs)

    return counts

# Streamlit code to create the interface
st.title("Steel Pipe Detector")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    counts = predict(image)
    st.write(f"Detected {counts} steel pipes.")
