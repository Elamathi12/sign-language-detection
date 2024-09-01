import cv2
import mediapipe as mp
import csv
import copy
import itertools
import string
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(letter, landmark_list):
    csv_path = 'keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter, *landmark_list])

# Define the alphabet and numbers
alphabet = list(string.ascii_uppercase) + ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# Directory containing the dataset
address = 'D:/Mathi/GITHUB/sign-language-detection/dataset/Indian/'
IMAGE_FILES = []

# Collect all image paths
for letter in alphabet:
    for j in range(1199):
        file_path = os.path.join(address, letter, f'{j}.jpg')
        IMAGE_FILES.append(file_path)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
    for idx, file in enumerate(IMAGE_FILES):
        print(f"Processing file: {file}")
        
        # Extract the letter from the file path (assuming directory name is the letter)
        letter = os.path.basename(os.path.dirname(file))
        print(f"Extracted letter: {letter}")
        
        # Read and process the image
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # If no hand landmarks are detected, skip this image
        if not results.multi_hand_landmarks:
            print(f"No hand landmarks detected in {file}")
            continue
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            logging_csv(letter, pre_processed_landmark_list)
            print(f"Logged keypoints for {letter} from {file}")

        print(f"Finished processing {file}\n")

    print("Keypoint generation complete.")
