from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time
import random
import threading
import logging
from logging.handlers import RotatingFileHandler

# Set up logging to file with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a rotating file handler (max 1MB, keep 5 backups)
handler = RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Remove console handler to prevent logging to stdout
logger.handlers = [handler]  # Ensure only file handler is used

app = Flask(__name__)

# Initialize webcam with DirectShow backend
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not video.isOpened():
    logger.error("Could not open webcam")
    raise Exception("Error: Could not open webcam")

# Shared frame for thread-safe access
latest_frame = None
frame_lock = threading.Lock()

# Load Haar cascades with error handling
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

if faceCascade.empty() or smileCascade.empty():
    logger.error("Failed to load Haar cascade files")
    raise Exception("Error: Could not load Haar cascade files")

# Global variables
total_images = 3
smile_frame_count = 0
required_frames = 15
images_captured = 0
folder_code = None
folder_path = None
video_feed_active = True  # Flag to control video feed

def generate_new_folder():
    global folder_code, folder_path
    folder_code = str(random.randint(100000, 999999))
    folder_path = os.path.join("static/images", folder_code)
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"Generated new folder: {folder_path}")
    return folder_code, folder_path

@app.route('/')
def index():
    global folder_code, folder_path, images_captured, video_feed_active
    folder_code, folder_path = generate_new_folder()
    images_captured = 0
    video_feed_active = True  # Reset video feed status
    logger.info("Initialized index route")
    return render_template('index.html', folder_code=folder_code)

def detect_smile():
    global images_captured, smile_frame_count, video_feed_active
    smile_history = []
    smile_confidences = []

    while images_captured < total_images and video_feed_active:
        success, img = video.read()
        if not success or img is None:
            logger.warning("Failed to read frame from webcam")
            continue

        # Store latest frame for /detection_status
        with frame_lock:
            global latest_frame
            latest_frame = img.copy()

        # Preprocess image
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayImg = clahe.apply(grayImg)

        # Detect faces
        faces = faceCascade.detectMultiScale(
            grayImg,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(30, 30)
        )

        smile_detected = False
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            roi_gray = grayImg[y:y+h, x:x+w]

            smiles = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.9,
                minNeighbors=25,
                minSize=(20, 10)
            )

            for (sx, sy, sw, sh) in smiles:
                smile_ratio = sw / sh
                smile_intensity = sw * sh

                min_smile_ratio = 1.5 if w > 100 else 1.6
                min_smile_intensity = 5000 if w > 100 else 6000

                if smile_ratio > min_smile_ratio and smile_intensity > min_smile_intensity:
                    smile_confidences.append(smile_intensity)
                    if len(smile_confidences) > 5:
                        smile_confidences.pop(0)
                    avg_confidence = sum(smile_confidences) / len(smile_confidences)

                    if avg_confidence > min_smile_intensity:
                        smile_history.append(1)
                        smile_detected = True
                        if len(smile_history) > 15:
                            smile_history.pop(0)

                        if len(smile_history) == 15 and sum(smile_history) >= 10:
                            img_path = os.path.join(folder_path, f"image_{images_captured+1}.jpg")
                            cv2.imwrite(img_path, img)
                            images_captured += 1
                            smile_history = []
                            smile_confidences = []
                            logger.info(f"Captured image {images_captured}: {img_path}")
                            if images_captured < total_images:
                                time.sleep(2)
                else:
                    smile_history.append(0)
                    if len(smile_history) > 15:
                        smile_history.pop(0)

        if not smile_detected:
            smile_history.append(0)
            smile_confidences = []
            if len(smile_history) > 15:
                smile_history.pop(0)

        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            logger.warning("Failed to encode frame to JPEG")
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Log when video feed stops
    logger.info("Video feed stopped")

@app.route('/video_feed')
def video_feed():
    logger.info("Starting video feed")
    return Response(detect_smile(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_images')
def get_images():
    global folder_code, images_captured, video_feed_active
    if folder_code is None:
        logger.warning("folder_code is None, initializing new folder")
        folder_code, folder_path = generate_new_folder()
        images_captured = 0
    video_feed_active = False  # Stop video feed when viewing images
    logger.info(f"Returning {images_captured} images for folder {folder_code}")
    return jsonify({"images": [f"{folder_code}/image_{i+1}.jpg" for i in range(images_captured)]})

@app.route('/detection_status')
def detection_status():
    global video_feed_active
    with frame_lock:
        if latest_frame is None:
            logger.warning("No latest frame available")
            return jsonify({"message": "Camera error! Check webcam connection."})
        img = latest_frame.copy()
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) == 0:
        logger.info("No faces detected")
        return jsonify({"message": "No face detected! Please move closer or adjust lighting."})
    
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    roi_gray = grayImg[y:y+h, x:x+w]
    smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.9, minNeighbors=25, minSize=(20, 10))
    
    if len(smiles) > 0:
        logger.info("Smile detected")
        return jsonify({"message": "Smile detected! Keep smiling..."})
    logger.info("No smile detected")
    return jsonify({"message": "Smile to capture! 😄"})

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)  # Disable reloader to avoid socket error
    finally:
        logger.info("Releasing webcam")
        video.release()
        cv2.destroyAllWindows()