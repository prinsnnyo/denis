import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_mcs_mouth.xml')
body_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_fullbody.xml')
upper_body_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_upperbody.xml')  

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_frame(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyeglasses_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        if not eyeglasses_cascade.empty():
            eyes_with_glasses = eyeglasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes_with_glasses:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)  # Yellow for glasses

        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            if my > h // 2:  
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

  
    # bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 100))
    # for (bx, by, bw, bh) in bodies:
    #     cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

    # Detect upper body
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (ux, uy, uw, uh) in upper_bodies:
        cv2.rectangle(frame, (ux, uy), (ux + uw, uy + uh), (0, 255, 255), 2)

    return frame

def process_image(image_path):
    
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    
    # Resize the image to improve detection
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)  # Scale the image by 1.5x
    
    processed_frame = process_frame(frame)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(processed_path, processed_frame)
    return processed_path

def process_video(video_path):
   
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

def gen_camera():
    """Generate frames from the webcam for real-time tracking."""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
   
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if the file is a video or image
        if filename.rsplit('.', 1)[1].lower() in {'mp4'}:
            return Response(process_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            processed_path = process_image(filepath)
            if processed_path:
                return send_file(processed_path, mimetype='image/jpeg')
            else:
                return "Error processing the file.", 400
    return "Invalid file type.", 400

@app.route('/camera')
def camera():
    """Stream real-time camera tracking."""
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)