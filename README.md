# Hand-Gesture-Controll-Presentation-and-Virtual-Quiz

This project is a hand gesture-controlled system designed to manage presentations and quizzes using a webcam. The script leverages computer vision to detect gestures and maps them to interactive actions like navigating slides, selecting quiz answers, and annotating slides.

## Features

### 1. Hand Gesture Detection
- Detects and tracks hand gestures using `cvzone.HandTrackingModule`.
- Maps specific gestures to actions for controlling presentations and quizzes.

### 2. Presentation Control
- Navigate slides forward and backward with simple hand gestures.
- Annotate slides by drawing directly on them using finger movements.

### 3. Quiz Interaction
- Presents multiple-choice questions.
- Detects gestures to select answers and provides feedback on correctness.
- Tracks and displays the user's score.

### 4. Mode Switching
- Seamlessly switch between Presentation and Quiz modes using a predefined gesture.

## Prerequisites

### Required Libraries
- OpenCV
- cvzone
- os
- numpy

To install the required libraries, run:
```bash
pip install opencv-python cvzone numpy
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/hand-gesture-control.git
   cd hand-gesture-control
   ```
2. Add your presentation slides as image files in the `Presentation` folder.
3. Edit the `quiz.json` file to add your quiz questions and answers (if applicable).

## How It Works

### Presentation Mode
1. **Navigation:** Use gestures to go to the next or previous slide.
2. **Annotation:** Activate drawing mode to write or draw directly on the slides using your finger.

### Quiz Mode
1. **Question Display:** The script shows one question at a time with answer options.
2. **Answer Selection:** Use hand gestures corresponding to options (A, B, C, or D).
3. **Feedback:** The system validates the answer and updates your score.

### Gesture Mappings
  | Gesture               | Action              |
  |-----------------------|---------------------|
  | Swipe Right           | Next Slide         |
  | Swipe Left            | Previous Slide     |
  | Two Fingers Up        | Enable Drawing     |
  | Closed Hand           | Erase Drawing      |
  | Hand Open (Palm)      | Switch Mode        |
  | Specific Finger Combo | Quiz Answer Select |
  
## Usage
  
  1. Run the script:
     ```bash
     python main.py
     ```
  2. Ensure your webcam is functional.
  3. Use the gestures to control the presentation or interact with the quiz.
  
  ## File Structure
  ```plaintext
  hand-gesture-control/
  ├── main.py          # Main script
  ├── Presentation/    # Folder for slide images
  ├── quiz.json        # Quiz questions and answers
  ├── README.md        # Project documentation
```

## Future Enhancements

- **Error Handling:** Improve robustness for missing slides or quiz files.
- **Dynamic Gesture Mapping:** Allow customizable gesture configurations.
- **Extensibility:** Add support for `.pptx` files or external multimedia.
- **UI Improvements:** Display visual feedback and summary screens.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to contribute to this project by submitting issues or pull requests!

