import cv2
import os
import json
import numpy as np

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure dataset directory exists
DATASET_DIR = "dataSet"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def save_user_details(user_id, user_name):
    """Save user ID and name mapping in names.json"""
    file_path = "names.json"
    data = {}

    # Load existing data if available
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    # Add or update user details
    data[str(user_id)] = user_name  # Ensure ID is stored as a string

    # Save updated data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def apply_clahe(image):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance face images"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def align_face(image, bbox):
    """Crop the face only, no alignment"""
    x, y, w, h = bbox
    face_roi = image[y:y+h, x:x+w]
    return face_roi

def create_dataset(user_id, user_name, num_samples=400):
    """Capture images and store user details with enhanced accuracy"""
    cap = cv2.VideoCapture(0)
    sample_num = 0

    save_user_details(user_id, user_name)  # Save user info

    while sample_num < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

        for (x, y, w, h) in faces:
            sample_num += 1

            # Just crop the face
            cropped_face = align_face(frame, (x, y, w, h))
            
            # Convert to grayscale and apply CLAHE
            gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            enhanced_face = apply_clahe(gray_face)

            # Save images (Original & Preprocessed)
            img_path_gray = f"{DATASET_DIR}/User.{user_id}.{sample_num}.jpg"
            img_path_color = f"{DATASET_DIR}/User.{user_id}.{sample_num}_color.jpg"
            cv2.imwrite(img_path_gray, enhanced_face)  # Save enhanced grayscale
            cv2.imwrite(img_path_color, cropped_face)  # Save cropped color image

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {sample_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.waitKey(100)  # Dynamic delay based on capture progress

        cv2.imshow("Face Capture", frame)
        cv2.waitKey(1)

        if sample_num >= num_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Dataset for {user_name} (ID: {user_id}) collected successfully!")

if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")
    create_dataset(user_id, user_name)
