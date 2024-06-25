import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle

data_dir = 'C:/Users/User/Downloads/sign-language-detector-python-master/sign-language-detector-python-master/data/'

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

# Iterate over each file in the data directory
for img_path in os.listdir(data_dir):
    img_full_path = os.path.join(data_dir, img_path)
    
    if os.path.isfile(img_full_path):  # Ensure it's a file
        print(f"Processing image: {img_full_path}")
        img = cv2.imread(img_full_path)
        
        # Check if the image is successfully read
        if img is not None:
            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(img_rgb)
            
            # Check if any hands are detected
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(img_path)  # Using the image path as the label

                # Draw landmarks on the image
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Display the image
            plt.figure()
            plt.imshow(img_rgb)
            plt.title(f"Image: {img_path}")
        else:
            print(f"Failed to read image: {img_full_path}")
    else:
        print(f"Skipped non-file entry: {img_full_path}")

# Save processed data to a pickle file
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Processed data and labels saved to {output_file}")

# Display all processed images
plt.show()

# Clean up
hands.close()
