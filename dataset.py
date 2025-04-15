import cv2
import json
import os
import numpy as np

# Load pre-trained Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

DATASET_DIR = "dataSet"
os.makedirs(DATASET_DIR, exist_ok=True)

def save_user_details(user_id, user_name):
    file_path = "names.json"
    data = {}

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

    if user_id in data:
        print(f"⚠ ID {user_id} already exists with name '{data[user_id]}'.")
        choice = input("Do you want to overwrite the data? (y/n): ")
        if choice.lower() != 'y':
            print("❌ Operation cancelled.")
            return False

    data[user_id] = user_name

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    return True

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def align_face(image, face_coords):
    x, y, w, h = face_coords
    face_roi = image[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face)

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye, right_eye = eyes[:2]

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(face_roi, rotation_matrix, (w, h))

        return aligned_face

    return face_roi

def create_dataset(user_id, user_name, num_samples=100):
    if not user_id.isdigit():
        print("❌ Invalid ID: Must be a number.")
        return

    if not save_user_details(user_id, user_name):
        return

    cap = cv2.VideoCapture(0)
    sample_num = 0

    print(f"📸 Starting image capture for {user_name} (ID: {user_id})...")

    while sample_num < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not working.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

        for (x, y, w, h) in faces:
            sample_num += 1

            aligned_face = align_face(frame, (x, y, w, h))
            gray_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            enhanced_face = apply_clahe(gray_face)

            img_path_gray = f"{DATASET_DIR}/User.{user_id}.{sample_num}.jpg"
            img_path_color = f"{DATASET_DIR}/User.{user_id}.{sample_num}_color.jpg"

            cv2.imwrite(img_path_gray, enhanced_face)
            cv2.imwrite(img_path_color, aligned_face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {sample_num}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            cv2.waitKey(100)

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Dataset for {user_name} (ID: {user_id}) collected successfully!")

# Main
if __name__ == "__main__":
    user_id = input("Enter Student ID (numbers only): ")
    user_name = input("Enter Student Name: ")
    create_dataset(user_id, user_name)
