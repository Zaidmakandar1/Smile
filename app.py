from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time
import random

app = Flask(__name__)
video = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

total_images = 3  
smile_frame_count = 0  
required_frames = 15  

# Generate new folder each time
def generate_new_folder():
    folder_code = str(random.randint(100000, 999999))
    folder_path = os.path.join("static/images", folder_code)
    os.makedirs(folder_path, exist_ok=True)
    return folder_code, folder_path

@app.route('/')
def index():
    global folder_code, folder_path, images_captured
    folder_code, folder_path = generate_new_folder()  # Regenerate folder
    images_captured = 0  # Reset captured images count
    return render_template('index.html', folder_code=folder_code)

def detect_smile():
    global images_captured, smile_frame_count
    while images_captured < total_images:
        success, img = video.read()
        if not success:
            break

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=6)

        for (x, y, w, h) in faces:
            roi_gray = grayImg[y:y+h, x:x+w]
            smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.96, minNeighbors=33)

            for (sx, sy, sw, sh) in smiles:
                smile_ratio = sw / sh  
                smile_intensity = sw * sh  

                if smile_ratio > 1.6 and smile_intensity > 6000:
                    smile_frame_count += 1
                    if smile_frame_count >= required_frames:
                        img_path = os.path.join(folder_path, f"image_{images_captured+1}.jpg")
                        cv2.imwrite(img_path, img)
                        images_captured += 1
                        smile_frame_count = 0  
                        if images_captured < total_images:
                            time.sleep(2)
                else:
                    smile_frame_count = 0  

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_smile(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_images')
def get_images():
    return jsonify({"images": [f"{folder_code}/image_{i+1}.jpg" for i in range(images_captured)]})

if __name__ == '__main__':
    app.run(debug=True)
