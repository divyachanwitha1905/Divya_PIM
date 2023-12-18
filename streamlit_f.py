# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:01:12 2023

@author: DELL
"""


import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import requests
from ultralytics import YOLO
import gdown
import os

# Function to download the model file
def download_file(url, filename):
    gdown.download(url, filename, quiet=False)

# Replace 'direct_download_link' with your direct download link
download_file('https://colab.research.google.com/drive/1_WRWfTkg8swCDWVQPeOH4XX-m8dM4wnJ?usp=sharing', 'best.pt')

# Check if the model file exists and is a valid PyTorch model file
if os.path.exists('best.pt'):
    try:
        torch.load('best.pt')
        st.write("Model file is valid.")
    except Exception as e:
        st.write(f"Error loading model file: {e}")
else:
    st.write("Model file not found.")

# Load the model
model = YOLO('best.pt')

# Rest of your Streamlit code...



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
        results = model(image)

    # Process the outputs
    threshold = 0.5
    outputs = results.pred[0]  # Get the predictions
    outputs = [output for output in outputs if output[4] > threshold]

    # Count the number of objects detected
    counts = len(outputs)

    return counts, outputs

# Streamlit code to create the interface
st.title("Steel Pipe Detector")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    counts, outputs = predict(image)
    st.write(f"Detected {counts} steel pipes.")
    st.write(f"Labels: {outputs}")
