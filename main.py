import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from pptx.util import Inches
from pptx import Presentation
import os
import time

# Parameters
width, height = 1280, 720
gestureThreshold = 360
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables for PowerPoint Presentation Control
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Variables for Quiz
quiz_running = False
current_question_index = 0
score = 0
last_gesture_time = 0
gesture_cooldown = 2  # seconds
quiz = [
    {
        'question': 'What is Machine Learning?',
        'options': [
            'A) A method of data analysis that automates analytical model building.',
            'B) A technique for writing computer programs.',
            'C) A way to improve hardware performance.',
            'D) A type of human intelligence.'
        ],
        'answer': 'A'
    },
    {
        'question': 'Which of the following is a type of supervised learning?',
        'options': ['A) Clustering', 'B) Regression', 'C) Association', 'D) Dimensionality reduction'],
        'answer': 'B'
    },
    {
        'question': 'What is overfitting in machine learning?',
        'options': [
            'A) When a model performs well on new, unseen data.',
            'B) When a model performs well on training data but poorly on new data.',
            'C) When a model performs poorly on both training and new data.',
            'D) When a model is too simple to capture the underlying patterns.'
        ],
        'answer': 'B'
    },
    {
        'question': 'Which of the following is a commonly used algorithm for classification?',
        'options': ['A) K-means', 'B) Linear Regression', 'C) Decision Tree', 'D) Principal Component Analysis (PCA)'],
        'answer': 'C'
    },
    {
        'question': 'What is the purpose of cross-validation in machine learning?',
        'options': [
            'A) To improve the model\'s training time.',
            'B) To reduce the model\'s complexity.',
            'C) To assess the model\'s performance on an independent dataset.',
            'D) To enhance the model\'s interpretability.'
        ],
        'answer': 'C'
    },
    {
        'question': 'What is a confusion matrix used for in machine learning?',
        'options': [
            'A) To visualize the performance of a classification algorithm.',
            'B) To measure the error rate of a regression model.',
            'C) To assess the complexity of a clustering algorithm.',
            'D) To determine the optimal number of features.'
        ],
        'answer': 'A'
    },
    {
        'question': 'Which of the following is an example of unsupervised learning?',
        'options': ['A) Linear Regression', 'B) K-means Clustering', 'C) Support Vector Machines (SVM)', 'D) Logistic Regression'],
        'answer': 'B'
    },
    {
        'question': 'In machine learning, what is \'feature engineering\'?',
        'options': [
            'A) The process of training the model.',
            'B) The process of selecting, modifying, and creating new features.',
            'C) The process of evaluating the model.',
            'D) The process of deploying the model.'
        ],
        'answer': 'B'
    },
    {
        'question': 'What is the main goal of a regression algorithm?',
        'options': [
            'A) To classify data points into predefined categories.',
            'B) To reduce the dimensionality of the dataset.',
            'C) To predict a continuous output variable.',
            'D) To group similar data points together.'
        ],
        'answer': 'C'
    },
    {
        'question': 'Which of the following techniques can be used to prevent overfitting?',
        'options': ['A) Increasing the model\'s complexity', 'B) Using more features', 'C) Cross-validation', 'D) Using a smaller dataset'],
        'answer': 'C'
    }
]

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)

# Function to display question on the screen
def display_question(question_data):
    question = question_data['question']
    options = question_data['options']

    # Create a white image for the quiz slide
    quiz_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 2
    line_type = cv2.LINE_AA

    # Add question and options
    y0, dy = 50, 40
    cv2.putText(quiz_img, question, (50, y0), font, font_scale, font_color, thickness, line_type)
    for i, option in enumerate(options):
        y = y0 + (i + 1) * dy
        cv2.putText(quiz_img, option, (50, y), font, font_scale, font_color, thickness, line_type)

    return quiz_img

# Function to handle quiz interaction
def handle_quiz_interaction(fingers):
    global current_question_index, score, last_gesture_time

    # Define the finger gestures for each option
    options_gestures = {
        'A': [0, 1, 0, 0, 0],  # Only index finger up
        'B': [0, 1, 1, 0, 0],  # Index and middle fingers up
        'C': [0, 1, 1, 1, 0],  # Index, middle, and ring fingers up
        'D': [0, 1, 1, 1, 1]   # All fingers up except thumb
    }

    for option, gesture in options_gestures.items():
        if fingers == gesture and time.time() - last_gesture_time > gesture_cooldown:
            last_gesture_time = time.time()
            if quiz[current_question_index]['answer'] == option:
                score += 1
            next_question()
            break

# Function to move to the next question
def next_question():
    global current_question_index, quiz_running

    if current_question_index + 1 < len(quiz):
        current_question_index += 1
    else:
        end_quiz()

# Function to end the quiz and show the score
def end_quiz():
    global quiz_running

    quiz_running = False
    print(f"Quiz Completed! Your score: {score}/{len(quiz)}")
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main loop
while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    fingers = []  # Initialize fingers to an empty list

    # Determine the mode (presentation or quiz)
    if quiz_running:
        # Quiz Mode
        hands, img = detectorHand.findHands(img)  # with draw
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]  # List of 21 Landmark points
            fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

            if quiz_running and current_question_index < len(quiz):
                handle_quiz_interaction(fingers)
                quiz_img = display_question(quiz[current_question_index])
                cv2.imshow("Quiz", quiz_img)

    else:
        # Presentation Mode
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        if imgCurrent is None:
            print(f"Image at {pathFullImage} not found.")
            continue

        hands, img = detectorHand.findHands(img)  # with draw
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands and buttonPressed is False:  # If hand is detected
            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]  # List of 21 Landmark points
            fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

            # Constrain values for easier drawing
            xVal = int(np.interp(lmList[8][0], [0, width], [0, imgCurrent.shape[1]]))
            yVal = int(np.interp(lmList[8][1], [0, height], [0, imgCurrent.shape[0]]))
            indexFinger = xVal, yVal

            if cy <= gestureThreshold:  # If hand is at the height of the face
                # Gesture 1 - Go to previous slide
                if fingers == [1, 0, 0, 0, 0]:
                    print("Left")
                    buttonPressed = True
                    if imgNumber > 0:
                        imgNumber -= 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
                # Gesture 2 - Go to next slide
                if fingers == [0, 0, 0, 0, 1]:
                    print("Right")
                    buttonPressed = True
                    if imgNumber < len(pathImages) - 1:
                        imgNumber += 1
                        annotations = [[]]
                        annotationNumber = -1
                        annotationStart = False
            # Gesture 3 - Show Pointer
            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            # Gesture 4 - Draw Pointer
            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                print(annotationNumber)
                annotations[annotationNumber].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            else:
                annotationStart = False
            # Gesture 5 - Erase Drawings
            if fingers == [0, 1, 1, 1, 1]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

        else:
            annotationStart = False

        if buttonPressed:
            counter += 1
            if counter > delay:
                counter = 0
                buttonPressed = False

        for i, annotation in enumerate(annotations):
            for j in range(len(annotation)):
                if j != 0:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

        # Adding Webcam Image on the Slide
        imgSmall = cv2.resize(img, (ws, hs))
        h, w, _ = imgCurrent.shape
        imgCurrent[0:hs, w - ws:w] = imgSmall

        cv2.imshow("Slides", imgCurrent)

        # Mode Switching Gesture
        if fingers == [1, 0, 0, 0, 1]:
            quiz_running = True
            print("Switched to Quiz Mode")

    # Window Closing Gesture
    if fingers == [1, 1, 1, 1, 1]:
        cv2.destroyAllWindows()
        print("All windows are closed")
        break

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

