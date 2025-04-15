import cv2
import numpy as np
import os
import json

# Define paths
dataset_path = "dataSet"  # Updated to match enhanced dataset script
recognizer_file = "recognizer.yml"
names_file = "names.json"

# Initialize recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load stored user names
if os.path.exists(names_file):
    with open(names_file, "r") as f:
        try:
            name_map = json.load(f)
        except json.JSONDecodeError:
            name_map = {}
else:
    name_map = {}

# Prepare training data
faces = []
ids = []

image_paths = sorted([
    os.path.join(dataset_path, f)
    for f in os.listdir(dataset_path)
    if f.endswith(".jpg") and not f.endswith("_color.jpg")  # Use only CLAHE grayscale images
])

for image_path in image_paths:
    filename = os.path.basename(image_path)
    try:
        student_id = filename.split(".")[1]
    except IndexError:
        print(f"⚠ Skipping invalid file: {filename}")
        continue

    name = name_map.get(student_id, "Unknown")
    print(f"🧠 Processing {filename} - ID: {student_id}, Name: {name}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not read image: {filename}")
        continue

    detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face_region = img[y:y + h, x:x + w]
        faces.append(face_region)
        ids.append(int(student_id))

# Train recognizer
if faces:
    recognizer.train(faces, np.array(ids))
    recognizer.save(recognizer_file)

    with open(names_file, "w") as f:
        json.dump(name_map, f, indent=4)

    print("✅ Training complete! Model saved.")
else:
    print("⚠ No valid faces found for training.")
