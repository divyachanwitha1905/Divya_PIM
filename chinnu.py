# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:59:31 2023

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:19:34 2023

@author: DELL
"""

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from ultralytics import YOLO
import gdown
import os
import pandas as pd  # Import pandas for DataFrame operations
import torchvision.ops as ops

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

def draw_polygons(image, outputs):
    draw = ImageDraw.Draw(image)
    for output in outputs:
        # Get the bounding box coordinates
        boxes = output.boxes.xyxy  # Use the xyxy attribute
        for box in boxes:
            # Convert coordinates to integers
            coordinates = [int(coordinate) for coordinate in box]
            # Create a polygon from the bounding box coordinates
            polygon = [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[1]),
                       (coordinates[2], coordinates[3]), (coordinates[0], coordinates[3])]
            # Draw the polygon on the image
            draw.polygon(polygon, outline="red")
    return image

def non_max_suppression(boxes_df, iou_threshold):
    # Convert DataFrame to tensor
    boxes = torch.tensor(boxes_df[['xmin', 'ymin', 'xmax', 'ymax']].values)
    scores = torch.tensor(boxes_df['confidence'].values)

    # Apply NMS
    keep = ops.nms(boxes, scores, iou_threshold)

    # Return DataFrame with boxes that survived NMS
    return boxes_df.iloc[keep]



def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    results = model(image_tensor)  # Get model predictions

    # Process results
    detections = []
    for output in results.xyxy:  # Use 'xyxy' attribute
        if output is not None:
            for detection in output:  # Iterate over each detection
                x1, y1, x2, y2, conf, _, class_id = detection
                detections.append({
                    'xmin': x1.item(),
                    'ymin': y1.item(),
                    'xmax': x2.item(),
                    'ymax': y2.item(),
                    'confidence': conf.item(),
                    'class': int(class_id.item())
                })

    detections_df = pd.DataFrame(detections)

    # Apply confidence threshold
    detections_df = detections_df[detections_df['confidence'] > 0.5]

    # Apply non-maximum suppression
    detections_df = non_max_suppression(detections_df, iou_threshold=0.5)

    image_with_boxes = draw_polygons(image, detections_df)
    return len(detections_df), detections_df, image_with_boxes

# ...


    detections_df = pd.DataFrame(detections)

    # Apply confidence threshold
    detections_df = detections_df[detections_df['confidence'] > 0.5]

    # Apply non-maximum suppression
    detections_df = non_max_suppression(detections_df, iou_threshold=0.5)

    image_with_boxes = draw_polygons(image, detections_df)
    return len(detections_df), detections_df, image_with_boxes

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
