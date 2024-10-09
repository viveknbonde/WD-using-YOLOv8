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
import pygame
from datetime import datetime

# Set page config
st.set_page_config(page_title="Weapon Detection System", layout="wide")

# Email configuration (Gmail)
EMAIL_SENDER = "wdpai12@gmail.com"
EMAIL_PASSWORD = "xydi tqnp jaes icfm"  # Gmail App Password
EMAIL_RECEIVER = "vivekbonde29@gmail.com"

# Pushbullet configuration
PUSHBULLET_API_KEY = "o.96QOQw5doBIdfmAOpuiDYsQsaDl8Ar4o"  # Get this from pushbullet.com

# Alert cooldown time in seconds
ALERT_COOLDOWN = 300  # 5 minutes
last_alert_time = 0

# Initialize pygame for sound
pygame.mixer.init()

# Initialize session state for notifications and webcam
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Load the alert sound
@st.cache_resource
def load_alert_sound():
    sound = pygame.mixer.Sound("alert.mp3")  # Make sure to have an "alert.mp3" file in your project directory
    return sound

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
        
        st.sidebar.warning("Email alert sent!")
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
            st.sidebar.warning("Pushbullet alert sent!")
        else:
            st.sidebar.error(f"Failed to send Pushbullet alert: {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Failed to send Pushbullet alert: {str(e)}")

def play_sound_alert():
    alert_sound = load_alert_sound()
    alert_sound.play()

def add_notification(weapon_type):
    timestamp = datetime.now().strftime("%I:%M:%S %p")  # 12-hour time format
    notification = f"🚨 Weapon Detected ({weapon_type}) - Keep eye on situation ({timestamp})"
    st.session_state.notifications.insert(0, notification)  # Add new notification at the beginning of the list

def send_alerts(weapon_type):
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time >= ALERT_COOLDOWN:
        send_email_alert(weapon_type)
        send_pushbullet_alert(weapon_type)
        play_sound_alert()
        add_notification(weapon_type)
        last_alert_time = current_time

def check_for_weapons(results):
    weapon_classes = ['pistol', 'rifle', 'knife']  # Adjust based on your model's classes
    
    for detection in results.boxes.data.tolist():
        class_id = int(detection[5])
        class_name = results.names[class_id]
        confidence = detection[4]
        
        if class_name.lower() in weapon_classes and confidence >= 0.5:
            send_alerts(class_name)
            return True
    return False

def detect_objects(frame, model):
    results = model.predict(frame, conf=0.25)
    if check_for_weapons(results[0]):
        st.warning("⚠️ Weapon detected! Alerts have been sent.")
    return results[0]

def draw_results(frame, results):
    annotated_frame = results.plot()
    return annotated_frame

def preprocess_image(image):
    # Apply Gaussian blur for filtering (noise reduction)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the original A and B channels
    limg = cv2.merge((cl,a,b))
    
    # Convert image back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_uploaded_file(uploaded_file, model, placeholder):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension in ["jpg", "jpeg", "png"]:
        process_image(uploaded_file, model, placeholder)
    elif file_extension == "mp4":
        process_video(uploaded_file, model, placeholder)

def process_image(uploaded_file, model, placeholder):
    image = Image.open(uploaded_file)
    frame = np.array(image)
    preprocessed_frame = preprocess_image(frame)
    results = detect_objects(preprocessed_frame, model)
    annotated_frame = draw_results(frame, results)
    placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

def process_video(uploaded_file, model, placeholder):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_cap = cv2.VideoCapture(tfile.name)
    
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_image(frame)
        results = detect_objects(preprocessed_frame, model)
        annotated_frame = draw_results(frame, results)
        placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    video_cap.release()

def process_webcam(model, placeholder):
    cap = cv2.VideoCapture(0)
    
    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
        
        preprocessed_frame = preprocess_image(frame)
        results = detect_objects(preprocessed_frame, model)
        annotated_frame = draw_results(frame, results)
        placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    
    cap.release()

def process_rtsp_stream(rtsp_url, model, placeholder):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access RTSP stream")
            break
        preprocessed_frame = preprocess_image(frame)
        results = detect_objects(preprocessed_frame, model)
        annotated_frame = draw_results(frame, results)
        placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
    cap.release()

def display_notifications(placeholder):
    with placeholder:
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
            global EMAIL_RECEIVER, PUSHBULLET_API_KEY
            EMAIL_RECEIVER = st.text_input("Email Receiver", EMAIL_RECEIVER)
            PUSHBULLET_API_KEY = st.text_input("Pushbullet API Key", PUSHBULLET_API_KEY, type="password")
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