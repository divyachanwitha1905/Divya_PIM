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
from ultralytics import YOLO
import gdown
import os
import numpy as np
from PIL import Image

# Function to download the model file
def download_file(url, filename):
    try:
        gdown.download(url, filename, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

# Replace 'direct_download_link' with your direct download link
download_url = 'https://drive.google.com/uc?export=download&id=1rINJnXcNoDtRa8oLdEffy-YsfLOD_58i'
download_file(download_url, 'best.pt')



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

def draw_polygons(image, outputs):
    draw = ImageDraw.Draw(image)
    for output in outputs:
        # Get the bounding box coordinates
        boxes = output.boxes.xyxy  # Use the xyxy attribute
        for box in boxes:
            # Convert coordinates to integers
            coordinates = [int(coordinate) for coordinate in box]
            # Create a polygon from the bounding box coordinates
            polygon = [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]), (coordinates[2], coordinates[3]), (coordinates[0], coordinates[3])]
            # Draw the polygon on the image
            draw.polygon(polygon, outline="red")
    return image




def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    results = model(image_tensor)
    image_with_boxes = draw_polygons(image, results)
    return len(results), results, image_with_boxes







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

