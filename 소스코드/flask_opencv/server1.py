from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import cv2 as cv
import numpy as np
from model import KeyPointClassifier
import os
import mediapipe as mp


app = Flask(__name__)

video_camera = None
global_frame = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

"""
@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera(0)

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="녹화 시작")
    else:
        video_camera.stop_record()
        return jsonify(result="녹화 중지")
"""
@app.route('/video_stream')
def video_stream():
    global video_camera 
    global global_frame
    global image
    global image_width, image_height
    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame is not None:
            global_frame = frame
            frame_array = np.frombuffer(frame, np.uint8)
            image = cv.imdecode(frame_array, -1)
            if image is not None:
                image = cv.flip(image, 1)  # Mirror display
                debug_image = image.copy()
                image_width, image_height = image.shape[1], image.shape[0]  # Modified line
        else:
            frame_array = np.frombuffer(global_frame, np.uint8)
            image = cv.imdecode(frame_array, -1)
            if image is not None:
                debug_image = image.copy()
                image_width, image_height = image.shape[1], image.shape[0]  # Modified line

    # Rest of the code...

    # Rest of the code...

        """    
    while True:
        frame = video_camera.get_frame()

        if frame is not None:
            global_frame = frame
            image = cv.imdecode(np.frombuffer(frame, np.uint8), -1)
            image = cv.flip(image, 1)  # Mirror display
            debug_image = image.copy()
            image_width, image_height = frame.shape[1], frame.shape[0]  # Modified line

        else:
            image = cv.imdecode(np.frombuffer(global_frame, np.uint8), -1)
            debug_image = image.copy()
            image_width, image_height = global_frame.shape[1], global_frame.shape[0]  # Modified line
        """
        
        """
        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        """    
        cap_device = 0
        cap_width = 1920
        cap_height = 1080

        use_brect = True
        """
        # Camera preparation
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        """
        # Model load
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        keypoint_classifier = KeyPointClassifier()

        # Read labels
        with open('/Users/shhan/temp/Face-Emotion-Recognition-Package/flask_opencv/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = f.readlines()
            keypoint_classifier_labels = [label.strip() for label in keypoint_classifier_labels]

        # Load emojis
        emoji_folder = r'/Users/shhan/temp/Face-Emotion-Recognition-Package/flask_opencv/emojis'
        emojis = []
        for emoji_file in os.listdir(emoji_folder):
            emoji_path = os.path.join(emoji_folder, emoji_file)
            emoji = cv.imread(emoji_path, cv.IMREAD_UNCHANGED)
        emojis.append(emoji)

        mode = 0
        """
        # Create a named window with the desired dimensions
        cv.namedWindow('Facial Emotion Recognition', cv.WINDOW_NORMAL)
        cv.resizeWindow('Facial Emotion Recognition', video_camera.window_width, video_camera.window_height)
        """
        """
        while True:
        
            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
    
            # Camera capture
            ret, image = cap.read()
    
            if not ret:
                break
        """
        
        image = cv.flip(image, 1)  # Mirror display
        debug_image = image.copy()
        
        
        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Bounding box calculation
                brect = video_camera.calc_bounding_rect(debug_image)

                # Landmark calculation
                landmark_list = video_camera.calc_landmark_list(debug_image, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = video_camera.pre_process_landmark(landmark_list)

                # Emotion classification
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = video_camera.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = video_camera.draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])

                # Display emoji in a new window
                emoji_index = facial_emotion_id
                emoji = emojis[emoji_index]
                emoji_window_name = 'Emotion Emoji'
                cv.imshow(emoji_window_name, emoji)
                cv.moveWindow(emoji_window_name, video_camera.window_width + 20, 0)

        # Screen reflection
        cv.imshow('Facial Emotion Recognition', debug_image)    

"""
    cap.release()
    cv.destroyAllWindows()
"""

    


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)

