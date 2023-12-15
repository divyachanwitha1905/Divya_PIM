# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:13:03 2023

@author: DELL
"""


import subprocess

subprocess.check_call(["python", '-m', 'pip', 'install', 'ultralytics'])

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
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        results = model(image_tensor)

    # Draw bounding boxes on the image and get labels
    labels = []
    for output in results.xyxy[0]:
        # Get the coordinates
        x1, y1, x2, y2 = output[:4]
        # Draw the bounding box
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline ="red")
        # Get the label
        labels.append(output[-1])

    # Count the number of objects detected
    counts = len(results.xyxy[0])

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
