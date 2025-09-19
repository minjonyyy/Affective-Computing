import cv2 as cv
import numpy as np
from flask import Flask, render_template, Response, send_from_directory, render_template
import mediapipe as mp
from model import KeyPointClassifier
import os
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)
keypoint_classifier = KeyPointClassifier()

with open('/Users/shhan/temp/Face-Emotion-Recognition-Package/Facial-emotion-recognition-using-mediapipe-main/model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = f.readlines()
keypoint_classifier_labels = [label.strip() for label in keypoint_classifier_labels]

emoji_folder = '/Users/shhan/temp/Face-Emotion-Recognition-Package/flask_opencv/static/emojis'
emojis = []
for emoji_file in os.listdir(emoji_folder):
    emoji_path = os.path.join(emoji_folder, emoji_file)
    emoji = cv.imread(emoji_path, cv.IMREAD_UNCHANGED)
    emojis.append(emoji)

window_width = 600
window_height = 600

app = Flask(__name__)

cap_device = 0
cap_width = 1920
cap_height = 1080

use_brect = True

cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

emotion_stats = {
    "Angry": 0,
    "Happy": 0,
    "Sad": 0,
    "Surprise": 0,
    "Neutral": 0
}


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []


    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()


    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = landmark_point[0] - base_x
        temp_landmark_list[index][1] = landmark_point[1] - base_y


    temp_landmark_list = [coord for landmark_point in temp_landmark_list for coord in landmark_point]


    max_value = max(map(abs, temp_landmark_list))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    emoji_height = int(brect[3] - brect[1])
    text_y = brect[1] - emoji_height - 10
    cv.putText(image, facial_text, (brect[0], text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if facial_text != "":
        info_text = 'Emotion: ' + facial_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


        stats_text = 'Emotion Statistics:'
        cv.putText(image, stats_text, (brect[0] + 5, brect[1] - 26), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

        y = brect[1] - 52
        for emotion, count in emotion_stats.items():
            stats_text = f'{emotion}: {count}'
            cv.putText(image, stats_text, (brect[0] + 5, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
            y -= 22

        return image

def generate_frames():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv.flip(frame, 1)
        debug_frame = frame.copy()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                brect = calc_bounding_rect(debug_frame, face_landmarks)
                landmark_list = calc_landmark_list(debug_frame, face_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                emotion_label = keypoint_classifier_labels[facial_emotion_id]
                emotion_stats[emotion_label] += 1

                debug_frame = draw_bounding_rect(use_brect, debug_frame, brect)
                debug_frame = draw_info_text(debug_frame, brect, emotion_label)

        ret, buffer = cv.imencode('.jpg', debug_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@app.route('/')
def index():
    return render_template('index.html', emotion_stats=emotion_stats)

@app.route('/static/emojis/<path:path>')
def send_emoji(path):
    return send_from_directory('static/emojis', path, cache_timeout=0)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
