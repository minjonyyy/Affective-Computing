import cv2 as cv
import numpy as np
import os
import mediapipe as mp
from model import KeyPointClassifier

use_brect = True

# Function to resize the image
def resize_image(image, width, height):
    return cv.resize(image, (width, height))

window_width = 600
window_height = 600

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

cap_device = 0
cap_width = 1920
cap_height = 1080

use_brect = True


cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

keypoint_classifier = KeyPointClassifier()


with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = f.readlines()
keypoint_classifier_labels = [label.strip() for label in keypoint_classifier_labels]


emoji_folder = r'C:\Users\minjonyyy\Downloads\Facial-emotion-recognition-using-mediapipe-main\emojis'
emojis = []
for emoji_file in os.listdir(emoji_folder):
    emoji_path = os.path.join(emoji_folder, emoji_file)
    emoji = cv.imread(emoji_path, cv.IMREAD_UNCHANGED)
    emojis.append(emoji)

mode = 0


cv.namedWindow('Facial Emotion Recognition', cv.WINDOW_NORMAL)
cv.resizeWindow('Facial Emotion Recognition', window_width, window_height)

cv.namedWindow('Emotion Statistics', cv.WINDOW_NORMAL)
cv.resizeWindow('Emotion Statistics', window_width, window_height)

emotion_stats = {
    "Angry": 0,
    "Happy": 0,
    "Sad": 0,
    "Surprise": 0,
    "Neutral": 0
}

while True:

    key = cv.waitKey(10)
    if key == 27:  # ESC
        break


    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)
    debug_image = image.copy()


    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            brect = calc_bounding_rect(debug_image, face_landmarks)
            landmark_list = calc_landmark_list(debug_image, face_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            emotion_label = keypoint_classifier_labels[facial_emotion_id]
            emotion_stats[emotion_label] += 1


            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])


            emoji_index = facial_emotion_id
            emoji = emojis[emoji_index]
            emoji_window_name = 'Emotion Emoji'
            cv.imshow(emoji_window_name, emoji)
            cv.moveWindow(emoji_window_name, window_width + 20, 0)


            stats_window_name = 'Emotion Statistics'
            stats_image = np.zeros((window_height, window_width, 3), np.uint8)  # Black background


            stats_text = ''
            cv.putText(stats_image, stats_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)


            y = 100
            for emotion, count in emotion_stats.items():
                stats_text = f'{emotion}: {count}'
                cv.putText(stats_image, stats_text, (50, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
                           cv.LINE_AA)
                y += 30

            cv.putText(stats_image, stats_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            cv.imshow(stats_window_name, stats_image)

    cv.imshow('Facial Emotion Recognition', debug_image)
cap.release()
cv.destroyAllWindows()