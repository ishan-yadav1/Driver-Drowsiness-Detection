
# Driver Drowsiness Detection System

This project is a computer vision application designed to detect driver drowsiness. It can analyze both a live webcam feed and pre-recorded video files. By tracking the driver's eyes using the Eye Aspect Ratio (EAR) algorithm, it can identify signs of fatigue and trigger an audible alarm to help prevent accidents.

---

## Features

- **Real-Time Detection:** Monitors the driver's face and eyes using a live webcam stream.
- **Video File Analysis:** Can process pre-recorded video files to detect drowsiness.
- **Eye Aspect Ratio (EAR):** Uses the EAR algorithm to accurately determine the level of eye closure.
- **Audible Alarm:** Plays an alarm sound to alert the driver when drowsiness is detected for a sustained period.

---

## Technologies & Libraries Used

- Python
- OpenCV
- Dlib
- SciPy
- imutils

---

## Setup and Installation

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**
```bash

git clone https://github.com/ishan-yadav1/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

**2. Create & Activate a V

irtual Environment**

```bash

# First, create the environment
python -m venv venv

# Next, activate it. Choose the command for your operating system.

# On Windows:
venv\Scripts\activate

# On MacOS or Linux:
# source venv/bin/activate
```

**3. Install Dependencies**

```bash

pip install -r requirements.txt
```


**4. Download the Shape Predictor Model**
This project requires a pre-trained facial landmark detector from dlib.

- **Download the file:** [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- **Extract it:** Unzip the downloaded file to get `shape_predictor_68_face_landmarks.dat`.
- **Place it:** Make sure the `shape_predictor_68_face_landmarks.dat` file is in the main project folder.

---

## How to Use

To run the program, you need to execute the main script `result.py`.

**To run detection on a live webcam feed:**

1.  Open the `result.py` file.
2.  Make sure the line that sets the video source is configured for the webcam (e.g., `source = 0`).
3.  Run the script from your terminal:
    ```bash
    python result.py
    ```

**To run detection on a video file:**

1.  Place your video file (e.g., `my_test_video.avi`) inside the project folder.
2.  Open the `result.py` file.
3.  Find the line of code that sets the video source.
4.  Change the value to the name of your video file in quotes (e.g., `source = 'my_test_video.avi'`).
5.  Save the `result.py` file.
6.  Run the script from your terminal:
    ```bash
    python result.py
    ```
---


