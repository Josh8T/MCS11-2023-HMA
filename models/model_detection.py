import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import sklearn

import warnings
warnings.filterwarnings('ignore')

# Define helper functions
def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    # Determine important landmarks for plank
    landmarks = results.pose_landmarks.landmark

    # Drawing helpers
    mp_pose = mp.solutions.pose
    
    IMPORTANT_LMS = [
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]

    # Generate all columns of the data frame

    HEADERS = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def model_detection(cap, model_path, input_scaler_path, VIDEO_PATH=None):

    IMPORTANT_LMS = [
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]

    # Generate all columns of the data frame

    HEADERS = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


    # Drawing helpers
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Load model
    with open(model_path, "rb") as f:
        sklearn_model = pickle.load(f)

    # Dump input scaler
    with open(input_scaler_path, "rb") as f2:
        input_scaler = pickle.load(f2)

    if VIDEO_PATH is None:
        cap = cv2.VideoCapture(cap)
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        
    current_stage = ""
    prediction_probability_threshold = 0.6

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print(cap.isOpened())
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break

            # Reduce size of a frame
            image = rescale_frame(image, 100)
            # image = cv2.flip(image, 1)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                print("No human found")
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Make detection
            try:
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                predicted_class = sklearn_model.predict(X)[0]
                prediction_probability = sklearn_model.predict_proba(X)[0]
                print(predicted_class, prediction_probability)

                # Evaluate model prediction
                # if predicted_class == "C" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                if predicted_class == 0 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                    current_stage = "Correct"
                # elif predicted_class == "L" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                elif predicted_class == 2 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    current_stage = "Low back"
                # elif predicted_class == "H" and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                elif predicted_class == 1 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    current_stage = "High back"
                else:
                    current_stage = "unk"
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probability
                cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prediction_probability[np.argmax(prediction_probability)], 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
            
            cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
        for i in range (1, 5):
            cv2.waitKey(1)
    

def main():
    model_detection(0)

if __name__ == "__main__":
    main()