import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import time
from datetime import datetime
from twilio.rest import Client
import os
from playsound import playsound
import threading

# Set page config
st.set_page_config(page_title="Weapon Detection System", layout="wide")

# Email configuration (Gmail)
EMAIL_SENDER = "wdpai12@gmail.com"
EMAIL_PASSWORD = "xydi tqnp jaes icfm"  # Gmail App Password
EMAIL_RECEIVER = "vivekbonde29@gmail.com"

# Pushbullet configuration
PUSHBULLET_API_KEY = "o.96QOQw5doBIdfmAOpuiDYsQsaDl8Ar4o"

# Twilio configuration
TWILIO_ACCOUNT_SID = "ACe270bfa36580165dccb65980349e77cc"
TWILIO_AUTH_TOKEN = "98c0796a3a11e4be0cc8b3a8df32da38"
TWILIO_PHONE_NUMBER = "+1 808 517 5810"
SMS_RECEIVER = "+918550932446"

# Alert cooldown time in seconds
ALERT_COOLDOWN = 300  # 5 minutes

# Alert sound file path
ALERT_SOUND_FILE = "alert.mp3"

# Initialize session state
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0

# Initialize the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best_yolov8.pt")

def send_email_alert(weapon_type):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = "ALERT: Weapon Detected!"
        
        body = f"A {weapon_type} has been detected by the surveillance system."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        st.sidebar.success("Email alert sent!")
    except Exception as e:
        st.sidebar.error(f"Failed to send email alert: {str(e)}")

def send_pushbullet_alert(weapon_type):
    try:
        data = {
            "type": "note",
            "title": "ALERT: Weapon Detected!",
            "body": f"A {weapon_type} has been detected by the surveillance system."
        }
        resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                            json=data,
                            headers={'Authorization': f'Bearer {PUSHBULLET_API_KEY}',
                                    'Content-Type': 'application/json'})
        if resp.status_code == 200:
            st.sidebar.success("Pushbullet alert sent!")
        else:
            st.sidebar.error(f"Failed to send Pushbullet alert: {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Failed to send Pushbullet alert: {str(e)}")

def send_sms_alert(weapon_type):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"ALERT: A {weapon_type} has been detected by the surveillance system.",
            from_=TWILIO_PHONE_NUMBER,
            to=SMS_RECEIVER
        )
        st.sidebar.success("SMS alert sent!")
    except Exception as e:
        st.sidebar.error(f"Failed to send SMS alert: {str(e)}")

def play_sound_alert():
    try:
        threading.Thread(target=playsound, args=(ALERT_SOUND_FILE,), daemon=True).start()
    except Exception as e:
        st.error(f"Failed to play sound alert: {str(e)}")

def add_notification(weapon_type):
    timestamp = datetime.now().strftime("%I:%M:%S %p")
    notification = f"🚨 Weapon Detected ({weapon_type}) - Keep eye on situation ({timestamp})"
    st.session_state.notifications.insert(0, notification)

def send_alerts(weapon_type):
    current_time = time.time()
    
    if current_time - st.session_state.last_alert_time >= ALERT_COOLDOWN:
        send_email_alert(weapon_type)
        send_pushbullet_alert(weapon_type)
        send_sms_alert(weapon_type)
        play_sound_alert()
        add_notification(weapon_type)
        st.session_state.last_alert_time = current_time
        return True
    return False

def check_for_weapons(results):
    weapon_classes = ['pistol', 'rifle', 'knife']
    
    for detection in results.boxes.data.tolist():
        class_id = int(detection[5])
        class_name = results.names[class_id]
        confidence = detection[4]
        
        if class_name.lower() in weapon_classes and confidence >= 0.5:
            if send_alerts(class_name):
                return True
    
    return False

def detect_objects(frame, model):
    results = model.predict(frame, conf=0.25)
    return results[0], check_for_weapons(results[0])

def draw_results(frame, results):
    annotated_frame = results.plot()
    return annotated_frame

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def process_uploaded_file(uploaded_file, model, main_placeholder):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension in ["jpg", "jpeg", "png"]:
        process_image(uploaded_file, model, main_placeholder)
    elif file_extension == "mp4":
        process_video(uploaded_file, model, main_placeholder)

def process_image(uploaded_file, model, main_placeholder):
    image = Image.open(uploaded_file)
    frame = np.array(image)
    preprocessed_frame = preprocess_image(frame)
    results, weapons_detected = detect_objects(preprocessed_frame, model)
    annotated_frame = draw_results(frame, results)
    main_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    if weapons_detected:
        st.warning("⚠️ Weapon detected! Alerts have been sent.")

def process_video(uploaded_file, model, main_placeholder):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_cap = cv2.VideoCapture(tfile.name)
    
    frame_count = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            preprocessed_frame = preprocess_image(frame)
            results, weapons_detected = detect_objects(preprocessed_frame, model)
            annotated_frame = draw_results(frame, results)
            main_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            if weapons_detected:
                st.warning("⚠️ Weapon detected! Alerts have been sent.")
    
    video_cap.release()

def process_webcam(model, main_placeholder):
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            preprocessed_frame = preprocess_image(frame)
            results, weapons_detected = detect_objects(preprocessed_frame, model)
            annotated_frame = draw_results(frame, results)
            main_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            if weapons_detected:
                st.warning("⚠️ Weapon detected! Alerts have been sent.")
    
    cap.release()

def process_rtsp_stream(rtsp_url, model, main_placeholder):
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access RTSP stream")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            preprocessed_frame = preprocess_image(frame)
            results, weapons_detected = detect_objects(preprocessed_frame, model)
            annotated_frame = draw_results(frame, results)
            main_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            if weapons_detected:
                st.warning("⚠️ Weapon detected! Alerts have been sent.")
    cap.release()

def display_notifications(placeholder):
    placeholder.empty()
    with placeholder.container():
        for notification in st.session_state.notifications:
            st.warning(notification)

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2C2C2C;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #2C2C2C;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Weapon Detection System")
    st.markdown("Real-time surveillance and alert system")

    # Sidebar for input selection and settings
    with st.sidebar:
        st.title("Controls")
        input_option = st.radio("Select Input Source", ["Upload Image/Video", "Webcam", "RTSP Stream"])
        
        if st.checkbox("Configure Alert Settings"):
            global EMAIL_RECEIVER, PUSHBULLET_API_KEY, SMS_RECEIVER
            EMAIL_RECEIVER = st.text_input("Email Receiver", EMAIL_RECEIVER)
            PUSHBULLET_API_KEY = st.text_input("Pushbullet API Key", PUSHBULLET_API_KEY, type="password")
            SMS_RECEIVER = st.text_input("SMS Receiver", SMS_RECEIVER)
            if st.button("Test Alerts"):
                send_alerts("test weapon")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection View")
        main_placeholder = st.empty()

    with col2:
        st.subheader("Notifications")
        notification_placeholder = st.empty()

    # Load the YOLOv8 model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return

    if input_option == "Upload Image/Video":
        uploaded_file = st.sidebar.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4"])
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file, model, main_placeholder)

    elif input_option == "Webcam":
        if st.sidebar.button("Start Webcam"):
            st.session_state.webcam_active = True
        if st.sidebar.button("Stop Webcam"):
            st.session_state.webcam_active = False
        
        if st.session_state.webcam_active:
            process_webcam(model, main_placeholder)

    elif input_option == "RTSP Stream":
        rtsp_url = st.sidebar.text_input("Enter RTSP URL", "rtsp://your_rtsp_stream_url")
        if st.sidebar.button("Start RTSP Stream"):
            process_rtsp_stream(rtsp_url, model, main_placeholder)

    # Display notifications
    display_notifications(notification_placeholder)

if __name__ == "__main__":
    main()
