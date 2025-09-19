import cv2 as cv
import threading
import numpy as np
import os
import mediapipe as mp
from model import KeyPointClassifier

class VideoCamera(object):
    
    def __init__(self):
        # Open a camera
        self.cap = cv.VideoCapture(0)
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()

        if ret:
            ret, jpeg = cv.imencode('.jpg', frame)

            # Record video
            # if self.is_record:
            #     if self.out == None:
            #         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #         self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))
                
            #     ret, frame = self.cap.read()
            #     if ret:
            #         self.out.write(frame)
            # else:
            #     if self.out != None:
            #         self.out.release()
            #         self.out = None  

            return jpeg.tobytes()
      
        else:
            return None
    
        # Function to resize the image
    def resize_image(image, width, height):
        return cv.resize(image, (width, height))

    window_width = 600
    window_height = 600

    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point


    def pre_process_landmark(landmark_list):
        temp_landmark_list = landmark_list.copy()

        # Convert to relative coordinates
        base_x, base_y = temp_landmark_list[0]
        for index, landmark_point in enumerate(temp_landmark_list):
            temp_landmark_list[index][0] = landmark_point[0] - base_x
            temp_landmark_list[index][1] = landmark_point[1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = [coord for landmark_point in temp_landmark_list for coord in landmark_point]

        # Normalization
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

        return image
"""
    cap_device = 0
    cap_width = 1920
    cap_height = 1080

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

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

    # Create a named window with the desired dimensions
    cv.namedWindow('Facial Emotion Recognition', cv.WINDOW_NORMAL)
    cv.resizeWindow('Facial Emotion Recognition', window_width, window_height)

    while True:
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
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
                brect = calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Emotion classification
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])

                # Display emoji in a new window
                emoji_index = facial_emotion_id
                emoji = emojis[emoji_index]
                emoji_window_name = 'Emotion Emoji'
                cv.imshow(emoji_window_name, emoji)
                cv.moveWindow(emoji_window_name, window_width + 20, 0)

        # Screen reflection
        cv.imshow('Facial Emotion Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()
"""

