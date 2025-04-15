import os
import json
import glob
import csv

dataset_path = "dataset"  # Folder where images are stored
names_file = "names.json"  # User data file
attendance_file_json = "attendance.json"  # Attendance stored in JSON format
attendance_file_csv = "attendance.csv"  # Attendance stored in CSV format

def delete_user(user_id):
    """Delete user details, images, and attendance records"""
    user_id = str(user_id)

    # Remove user from names.json
    if os.path.exists(names_file):
        with open(names_file, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}

        if user_id in data:
            print(f"Deleting user: {data[user_id]} (ID: {user_id})")
            del data[user_id]  # Remove user from JSON

            # Save updated data
            with open(names_file, "w") as file:
                json.dump(data, file, indent=4)

            print("User removed from names.json.")
        else:
            print("User ID not found in names.json.")
    
    # Delete user's images
    image_files = glob.glob(os.path.join(dataset_path, f"User.{user_id}.*.jpg"))
    for img in image_files:
        os.remove(img)
        print(f"Deleted: {img}")

    # Remove user from attendance.json (if exists)
    if os.path.exists(attendance_file_json):
        with open(attendance_file_json, "r") as file:
            try:
                attendance_data = json.load(file)
            except json.JSONDecodeError:
                attendance_data = {}

        if user_id in attendance_data:
            del attendance_data[user_id]  # Remove attendance entry

            with open(attendance_file_json, "w") as file:
                json.dump(attendance_data, file, indent=4)

            print("User removed from attendance.json.")
    
    # Remove user from attendance.csv (if exists)
    if os.path.exists(attendance_file_csv):
        updated_rows = []
        with open(attendance_file_csv, "r") as file:
            reader = csv.reader(file)
            header = next(reader)  # Read header

            for row in reader:
                if row[0] != user_id:  # Assuming user_id is in the first column
                    updated_rows.append(row)

        # Write updated attendance data
        with open(attendance_file_csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write header back
            writer.writerows(updated_rows)  # Write updated data

        print("User removed from attendance.csv.")

    print("User deletion process completed.")

if __name__ == "__main__":
    user_id = input("Enter User ID to delete: ")
    delete_user(user_id)
