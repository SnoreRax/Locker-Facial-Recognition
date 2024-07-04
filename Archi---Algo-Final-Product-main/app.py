import os
import cv2
import numpy as np
import requests
from flask import *

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Use a factory function to ensure a new instance is created
def create_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.FisherFaceRecognizer_create()

label_dict = {"Owner": 1, "Unknown": 2}
is_training = False
model_trained = False

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('training_images/owner'):
    os.makedirs('training_images/owner')

if not os.path.exists('training_images/unknown'):
    os.makedirs('training_images/unknown')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def prepare_training_data(folder_path):
    faces = []
    labels = []
    face_cascade = create_face_cascade()
    
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces_rect = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces_rect:
                face = image[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (240, 240))  # Resize the face to 240x240
                faces.append(face_resized)
                
                if "owner" in image_name:
                    labels.append(label_dict["Owner"])  # Label for the owner's face
                else:
                    labels.append(label_dict["Unknown"])  # Label for other known faces
    
    return faces, labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training_status')
def training_status():
    return jsonify({"model_trained": model_trained})

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files')
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            if "owner" in filename.lower():
                label = "Owner"
                save_path = os.path.join('training_images/owner', filename)
            elif "unknown" in filename.lower():
                label = "Unknown"
                save_path = os.path.join('training_images/unknown', filename)
            else:
                flash('File does not meet the naming criteria (must contain "owner" or "unknown")')
                return redirect(request.url)
            
            file.save(save_path)

    flash('Files successfully uploaded')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    global is_training, model_trained, recognizer
    if not os.listdir('training_images/owner') or not os.listdir('training_images/unknown'):
        flash('No training data available. Upload images first.')
        return redirect(url_for('index'))
    
    is_training = True
    faces, labels = prepare_training_data('training_images/owner')
    faces_unknown, labels_unknown = prepare_training_data('training_images/unknown')
    faces.extend(faces_unknown)
    labels.extend(labels_unknown)

    # Ensure there are at least two classes
    if len(set(labels)) < 2:
        flash("Error: At least two classes are needed to perform LDA. Please provide images for both 'owner' and 'unknown'.")
        return redirect(url_for('index'))

    recognizer.train(faces, np.array(labels))
    is_training = False
    model_trained = True
    flash('Model trained successfully')
    return redirect(url_for('index'))

def detect_and_recognize_face(frame, face_recognizer):
    face_cascade = create_face_cascade()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (240, 240))  # Resize the face to 240x240
        label, confidence = face_recognizer.predict(face_resized)
        
        print(f"Confidence: {confidence}, Label: {label}")

        #Code below checks for confidence and determines whether face detected is owner or unknown
        if label == label_dict["Owner"] and confidence > 100:  # Adjust confidence threshold if needed
            label_text = 'Owner'
            send_unlock_request()
        else:
            label_text = 'Unknown'

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

#Following method is to send an http request to the ESP32 to unlock the solenoid
def send_unlock_request():
    url = "http://10.15.25.231/unlock" #Use IP of ESP32 Web Server
    try:
        response = requests.get(url, timeout=5)
        print(response.text)
        if response.status_code == 200:
            print("Unlock request sent successfully!")
        else:
            print(f"Failed to send unlock request, status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending unlock request: {e}")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                response = requests.get('http://10.15.25.231/capture') #Use IP of ESP32 Web Server
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                frame = detect_and_recognize_face(frame, recognizer)
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            except Exception as e:
                print(f"Error generating frame: {e}")
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
