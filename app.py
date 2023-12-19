# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:19:34 2023

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:01:12 2023

@author: DELL
"""

import streamlit as st
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import requests
from ultralytics import YOLO
import gdown
import os
import numpy as np

# Function to download the model file
def download_file(url, filename):
    gdown.download(url, filename, quiet=False)

# Replace 'direct_download_link' with your direct download link
download_file('https://drive.google.com/uc?export=download&id=1rINJnXcNoDtRa8oLdEffy-YsfLOD_58i', 'best.pt')

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

# Define the transformation
transform = T.Compose([T.Resize(256),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def draw_boxes(image, outputs):
    # Create a draw object
    draw = ImageDraw.Draw(image)
    
    # Iterate over the outputs
    for output in outputs:
        print(f"Output: {output}")  # Add this line
        coordinates = [(output[i], output[i + 1]) for i in range(0, len(output[:4]), 2)]
        draw.polygon(coordinates, outline="red")
        draw.text(output[:2], output[4])
    
    return image




def predict(image):
    # Convert PIL Image to PyTorch Tensor
    image_tensor = transform(image).unsqueeze(0)
    
    # Perform prediction using the model
    results = model(image_tensor)
    
    # Check if results is a list or a similar iterable
    if isinstance(results, (list, tuple, set, np.ndarray)):
        # Process the results here
        counts = len(results)
        outputs = results
    else:
        print(f"Unexpected result type: {type(results)}")
        counts = 0
        outputs = []
    
    # Draw boxes on the original image
    image_with_boxes = draw_boxes(image, outputs)
    
    return counts, outputs, image_with_boxes

# Streamlit code to create the interface
st.title("Steel Pipe Detector")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Detecting...")
    counts, outputs, image_with_boxes = predict(image)
    st.image(image_with_boxes, caption='Detected Image.', use_column_width=True)
    st.write(f"Detected {counts} steel pipes.")
    st.write(f"Labels: {outputs}")
