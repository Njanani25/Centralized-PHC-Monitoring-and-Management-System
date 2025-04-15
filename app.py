from flask import Flask, flash, redirect, render_template, request, url_for,  session,jsonify
import cv2
import numpy as np
import os
import json
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import geocoder  # For getting location
import requests 
import csv
from twilio.rest import Client  # For sending SMS
import schedule
import time


app = Flask(__name__)
app.secret_key = "supersecretkey"

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "adpromain141@gmail.com"  # Replace with your email
SMTP_PASSWORD = "ofdp euil hwcg auqo"  # Replace with your app password
ADMIN_EMAIL = "adpromain141@gmail.com"  # Replace with the correct admin email

# Load Face Recognition Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained recognizer model
recognizer_file = "recognizer.yml"
if os.path.exists(recognizer_file):
    recognizer.read(recognizer_file)
else:
    print("Recognizer model not found. Train the model first.")


attendance_file = "attendance.csv"

# Ensure CSV exists with correct headers
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["ID", "Name", "Engagement", "Location", "Timestamp"])
    df.to_csv(attendance_file, index=False)

# Load student names
names_file = "names.json"
if os.path.exists(names_file):
    with open(names_file, "r") as file:
        name_data = json.load(file)
else:
    name_data = {}
#-------------------------location---------------------------------------------------
def get_location():
    """Fetch the current location of the student (City, Country)."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city", "Unknown")
        country = data.get("country", "Unknown")
        return f"{city}, {country}"
    except requests.RequestException:
        return "Unknown"
#---------------------------------------------send attedance------------------------------------------------
@app.route('/send-email', methods=['POST'])
def send_email():
    """Send attendance.csv file to the admin via email."""
    try:
        if not os.path.exists(attendance_file):
            return jsonify({"success": False, "message": "Attendance file not found!"})

        # Create Email
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = "📊 Daily Attendance Report"

        # Email Body
        body = "Please find the attached attendance report for today."
        msg.attach(MIMEText(body, 'plain'))

        # Attach the CSV File
        with open(attendance_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(attendance_file)}"')
            msg.attach(part)

        # Send Email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, ADMIN_EMAIL, msg.as_string())

        return jsonify({"success": True, "message": "Email sent successfully!"})

    except smtplib.SMTPAuthenticationError:
        return jsonify({"success": False, "message": "SMTP Authentication Error! Check email/password."})
    except smtplib.SMTPConnectError:
        return jsonify({"success": False, "message": "Could not connect to SMTP server."})
    except smtplib.SMTPRecipientsRefused:
        return jsonify({"success": False, "message": "Recipient email address refused!"})
    except smtplib.SMTPException as smtp_error:
        return jsonify({"success": False, "message": f"SMTP error: {smtp_error}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {e}"})
#--------------------------------mark attendance 18.3.25------------------------------------------------
def mark_attendance(student_id, student_name, engagement_level):
    """Mark attendance for detected students, updating the CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = get_location()
    df = pd.read_csv(attendance_file)
    
    student_mask = (df["ID"] == student_id) & (df["Timestamp"].str.startswith(datetime.today().strftime("%Y-%m-%d")))
    if not df[student_mask].empty:
        df.loc[student_mask, ["Engagement", "Timestamp", "Location"]] = [engagement_level, timestamp, location]
    else:
        new_entry = pd.DataFrame([[student_id, student_name, engagement_level, location, timestamp]],
                                 columns=["ID", "Name", "Engagement", "Location", "Timestamp"])
        df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    print(f"Attendance Updated: {student_name} (ID: {student_id}) - {engagement_level} - {location}")

#---------------------------------------face recognization----------------------------------------
def recognize_face():
    """Continuously detect faces and mark attendance."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access the camera.")
        return

    recognized_students = set()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("🔍 Face recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe_img = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(clahe_img, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = clahe_img[y:y + h, x:x + w]
            label = "Unknown"
            color = (0, 0, 255)

            try:
                student_id, confidence = recognizer.predict(face_roi)

                if confidence < 50 and str(student_id) in name_data:
                    student_name = name_data[str(student_id)]
                    label = f"{student_name} ({round(confidence, 2)})"
                    color = (0, 255, 0)

                    if student_id not in recognized_students:
                        mark_attendance(student_id, student_name, "Not Tracked")
                        recognized_students.add(student_id)

            except Exception as e:
                print(f"⚠ Error: {e}")

            # Display result
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    send_email()

#----------------------------------rendering main page---------------------------------
@app.route('/')
def index():
    return render_template("index.html")
#-------------------------------signup & login --------------------------------------------------
CSV_FILE = "login.csv"

# Ensure CSV exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["phc", "name", "email", "phone", "password"])  # Header
# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Signup Route
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    phc = data["phc"]
    name = data["name"]
    email = data["email"]
    phone = data["phone"]
    password = data["password"]

    # Check if email exists
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[2] == email:
                return "Email already exists! Please login."

    # Save new user
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([phc, name, email, phone, password])

    return "Signup successful! You can now log in."

# Login Route
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data["email"]
    password = data["password"]

    # Check if user exists
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[2] == email and row[4] == password:
                session["user"] = row[1]  # Store username in session
                return "success"  # Login successful

    return "Invalid email or password."
#------------------------------view page----------------------------------------------------
# View Page
@app.route("/view")
def view():
    if "user" in session:
        return render_template("view.html", username=session["user"])
    return redirect(url_for("home"))
#-------------------------------------detection---------------------------------------------------
@app.route('/detect', methods=["POST"])
def detect():
    recognize_face()
    flash("Attendance captured and emailed to admin.", "success")
    return redirect(url_for("index"))
#---------------------------------------------attendance-------------------------------------------
@app.route('/attendance')
def view_attendance():
    df = pd.read_csv(attendance_file)
    return render_template("attendance.html", tables=[df.to_html()], titles=df.columns.values) if not df.empty else "No attendance records found."


from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from statistics import mean


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SECRET_KEY'] = 'secretkey'

db = SQLAlchemy(app)


# Database Model for PHC Updates
class PHCUpdate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phc_name = db.Column(db.String(100), nullable=False)
    patient_count = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.Text, nullable=False)
    treatment = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Function to Send Email Alerts
def send_alert(email, message):
    msg = MIMEText(message)
    msg['Subject'] = "🚨 Healthcare Anomaly Alert"
    msg['From'] = SMTP_USERNAME
    msg['To'] = email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, email, msg.as_string())
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Route: Submit PHC Daily Update
@app.route('/submit_update', methods=['GET', 'POST'])
def submit_update():
    if request.method == 'POST':
        phc_name = request.form['phc_name']
        patient_count = int(request.form['patient_count'])
        diagnosis = request.form['diagnosis']
        treatment = request.form['treatment']

        new_entry = PHCUpdate(
            phc_name=phc_name,
            patient_count=patient_count,
            diagnosis=diagnosis,
            treatment=treatment
        )
        db.session.add(new_entry)
        db.session.commit()
        
        # Redirect to check for anomalies
        return redirect(url_for('check_alerts'))

    return render_template('submit_update.html')

#--------------------------anamoly=-----------------------------------------

from statistics import mean, stdev
from flask import flash, redirect, url_for

@app.route('/check_alerts')
def check_alerts():
    phc_names = db.session.query(PHCUpdate.phc_name).distinct().all()
    phc_names = [phc[0] for phc in phc_names]  # Extract PHC names from query result
    
    anomalies_detected = False

    for phc in phc_names:
        updates = PHCUpdate.query.filter_by(phc_name=phc).order_by(PHCUpdate.date.desc()).limit(7).all()
        patient_counts = [update.patient_count for update in updates]

        if len(patient_counts) < 2:
            flash(f"⚠️ Not enough data for {phc} to analyze trends.", "warning")
            continue  # Skip to the next PHC

        avg_count = mean(patient_counts[:-1])  # Exclude latest for average
        latest_count = patient_counts[-1]
        std_dev = stdev(patient_counts[:-1]) if len(patient_counts) > 2 else 0  # Standard deviation for better analysis

        # Check for a significant drop in patient count
        if latest_count < (avg_count * 0.5):  # 50% or more drop
            message = (f"🚨 Anomaly detected at {phc}! Sudden drop in patient count: "
                       f"{latest_count} (Expected: {avg_count:.2f})")
            send_alert(ADMIN_EMAIL, message)
            flash(message, "danger")
            anomalies_detected = True

        # Check for a significant surge in patient count
        elif latest_count > (avg_count * 1.5):  # 50% or more increase
            message = (f"⚠️ Possible outbreak at {phc}! Sudden rise in patient count: "
                       f"{latest_count} (Expected: {avg_count:.2f})")
            send_alert(ADMIN_EMAIL, message)
            flash(message, "warning")
            anomalies_detected = True

        # Check for high deviation in patient count
        elif std_dev > (avg_count * 0.3):  # If variation is high (30% of mean)
            message = (f"⚠️ Unusual fluctuations detected at {phc}! "
                       f"High variation in patient counts over the past week.")
            send_alert(ADMIN_EMAIL, message)
            flash(message, "warning")
            anomalies_detected = True

    if not anomalies_detected:
        flash("✅ No anomalies detected in any PHC.", "success")

    return redirect(url_for('dashboard'))

#-------------------------------------------dashboard csv file--------------------------------------------
def save_to_csv():
    """Exports PHCUpdate data to phcdetails.csv."""
    updates = PHCUpdate.query.order_by(PHCUpdate.date.desc()).all()
    file_path = "phcdetails.csv"
    
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "PHC Name", "Patient Count", "Diagnosis", "Treatment", "Date"])
        for update in updates:
            writer.writerow([update.id, update.phc_name, update.patient_count, update.diagnosis, update.treatment, update.date])

    return file_path

@app.route('/dashboard')
def dashboard():
    updates = PHCUpdate.query.order_by(PHCUpdate.date.desc()).all()
    phc_counts_dict = {update.phc_name: 0 for update in updates}

    for update in updates:
        phc_counts_dict[update.phc_name] += update.patient_count

    phc_labels = list(phc_counts_dict.keys())
    phc_counts = list(phc_counts_dict.values())

    return render_template('dashboard.html', updates=updates, phc_labels=phc_labels, phc_counts=phc_counts)

#------------------------------------dashboard send to mail-----------------------------------
phc_file = "phcdetails.csv"
@app.route('/phc-email', methods=['POST'])
def phc_email():
    """Send attendance.csv file to the admin via email."""
    try:
        if not os.path.exists(phc_file):
            return jsonify({"success": False, "message": "phc details file not found!"})

        # Create Email
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = "📊 PHC Report"

        # Email Body
        body = "Please find the attached PHC report for today."
        msg.attach(MIMEText(body, 'plain'))

        # Attach the CSV File
        with open(phc_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(phc_file)}"')
            msg.attach(part)

        # Send Email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, ADMIN_EMAIL, msg.as_string())

        return jsonify({"success": True, "message": "Email sent successfully!"})

    except smtplib.SMTPAuthenticationError:
        return jsonify({"success": False, "message": "SMTP Authentication Error! Check email/password."})
    except smtplib.SMTPConnectError:
        return jsonify({"success": False, "message": "Could not connect to SMTP server."})
    except smtplib.SMTPRecipientsRefused:
        return jsonify({"success": False, "message": "Recipient email address refused!"})
    except smtplib.SMTPException as smtp_error:
        return jsonify({"success": False, "message": f"SMTP error: {smtp_error}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {e}"})
#------------------------------------------send sms/email automated alert----------------------------

import threading
from datetime import datetime
# File paths
doctors_file = "doctors.csv"
attendance_file = "attendance.csv"

# Twilio Configuration
TWILIO_SID = "AC4db715ad671d87e0391073516737bca2"
TWILIO_AUTH_TOKEN = "d13ef9bffa8129dcf3bde2b6c0e7160b"
TWILIO_PHONE_NUMBER = "+19127159841"

# Email Configuration
SMTP_USERNAME = "adpromain141@gmail.com"
SMTP_PASSWORD = "ofdp euil hwcg auqo"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def check_doctor_attendance():
    """Check attendance and send alerts to absent doctors."""
    today = datetime.datetime.today().strftime("%Y-%m-%d")

    # Load doctor details
    doctors_df = pd.read_csv(doctors_file)

    # Load attendance records
    attendance_df = pd.read_csv(attendance_file)

    for _, doctor in doctors_df.iterrows():
        doctor_id = doctor["ID"]
        doctor_name = doctor["Name"].strip()
        email = doctor["Email"]
        phone = doctor["Phone"]

        print(f"👨‍⚕️ Checking {doctor_name} (ID: {doctor_id}) - Email: {email}, Phone: {phone}")

        # Check if the doctor has marked attendance today
        doctor_attendance = attendance_df[
            (attendance_df["ID"].astype(str) == str(doctor_id)) & 
            (attendance_df["Timestamp"].str.startswith(today))
        ]

        if doctor_attendance.empty:
            print(f"🚨 ALERT: {doctor_name} (ID: {doctor_id}) did NOT mark attendance. Sending email & SMS.")

            message = f"""
            Dear {doctor_name}, you have not marked attendance today.

            Please confirm your status:
            - [I will come](https://yourapp.com/confirm-attendance)
            - [I am on leave](https://yourapp.com/apply-leave)
            """

            send_alert_email(email, "Attendance Reminder", message)
            send_sms(phone, message)
        else:
            print(f"✅ {doctor_name} (ID: {doctor_id}) has already marked attendance.")

def send_alert_email(to_email, subject, message):
    """Send an email notification."""
    try:
        msg = MIMEText(message, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_USERNAME
        msg["To"] = to_email

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, to_email, msg.as_string())

        print(f"📧 Email sent successfully to {to_email}")
    except Exception as e:
        print(f"❌ Email error: {e}")

def send_sms(to_phone, message):
    """Send an SMS notification."""
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to="+919361895677"
        )
        print(f"📱 SMS sent successfully to {to_phone}")
    except Exception as e:
        print(f"❌ SMS error: {e}")

def run_scheduler():
    """Check the time every minute and run attendance check at 9 PM."""
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if timestamp  == "12:35":  # 9:00 PM
            print("📅 Running attendance check at 9 PM...")
            check_doctor_attendance()
        time.sleep(60)  # Wait a minute before checking again

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)


  

#recovery code for trillo- V9ECBESVTXCHUHJA35ULEGTG