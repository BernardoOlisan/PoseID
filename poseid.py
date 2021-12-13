'''
Real time pose detection, the password is going to be the pose

YOU CAN ONLY ENTER IF U MAKE THE SECRET POSE "T-POSE"
'''
import os
import cv2
import math
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark


# Defining Mediapipe vars
drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def detectPose(image, pose, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                               connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    return output_image, landmarks                 

# Function to calculate the angle of the body parts
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if angle is less than zero
    if angle < 0:
        angle += 360

    return angle


# Recognize the pose
def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0,0,255)

    # Calculate required angles
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Checking the pose "T-POSE"
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = 'T pose'

    if label != 'Unknown Pose':
        color = (0, 255, 0)

    # Labels output
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return output_image, label


# Initiate the webcam
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        
        image = cv2.flip(image, 1)

        image_height, image_width, _ = image.shape

        image = cv2.resize(image, (int(image_width * (640 / image_height)), 640))
        # pose landmarks detection
        image, landmarks = detectPose(image, pose, display=False)
        # Check if landmarks are detected
        if landmarks:
            # detect the pose
            image, _ = classifyPose(landmarks, image, display=False)

            if _ == "T pose":    
                image = cv2.putText(image, 'Bienvenido!', (200,500), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 10)

        cv2.imshow('PoseID', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

