# Face Recognition Project: Match Your Face with a U.S. President

This project uses face recognition technology to compare a user's face with images of U.S. Presidents.
It captures a photo using the user's webcam, processes the image, and then compares the scanned face with
images of the Presidents to find the best possible match using the face_recognition library and OpenCV (cv2)
to handle webcam input and image processing.

# Requirements
To run this project, you'll need the following:
-Python 3.x
-cv2 (OpenCV)
-numpy
-face_recognition

# Installation
You can install the required libraries using pip with the following command:
pip install opencv-python numpy face_recognition

# Project Files
The project files include the following:
Presidents.py: Contains the functions that handle face scanning, encoding, and matching the user's face
with a U.S. President.

main.py: The main script prompts the user to scan their face, processes it, and tells you the closest matching
President, as well as a similarity percentage. Also, the main script will display the scanned user image
and the matching President's image with box around each face.

test_Presidents.py: Unit tests for the functions in Presidents.py to test various cases.

# Usage
To note, the two directories ImagesPresidents and ImagesTest contain images that are used in Presidents.py and
test_Presidents.py respectively. To run the program, simply type python main.py and the program will begin.
Upon starting, the program will prompt you to give permission for using the webcam. It will then scan your face
and match it against the images of the presidents. The program will then display the president whose face matches
the most with yours, along with the similarity percentage.

If you wish to run the tests written for this program, you can run them using pytest by entering:
pytest test_Presidents.py