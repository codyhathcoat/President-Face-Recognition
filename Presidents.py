import cv2
import numpy as np
import face_recognition
import os
from typing import List, Tuple

'''
The purpose of this function is to scan a face using the systems camera. 
This happens using OpenCV to video capture and scan for a face. This means the function
will not return until a face has been scanned. Once a face has been scanned, it is properly 
encoded and returned for further use, along with the images RGB value.
'''
def get_face() -> Tuple[np.ndarray, List[np.ndarray]]:
    # Create the video capture object.
    cap = cv2.VideoCapture(0)

    # Raise a value error if the object can't be created.
    if not cap.isOpened():
        raise ValueError("Can't access the webcam.")

    did_scan = False  # Has a face been scanned yet.
    counter = 0

    # while did_scan is False and counter < 100:
    while did_scan == False and counter < 100:
        success, img = cap.read()  # Read from the camera.

        # If no frame could be captured.
        if not success:
            print("Failed to capture frame.")
            counter += 1
            # continue
            break

        # If a frame was captured .
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find the face location
        faces_cur_frame = face_recognition.face_locations(img_rgb)
        # Get the encoding for the face
        encodes_cur_frame = face_recognition.face_encodings(img_rgb, faces_cur_frame)
        #encode = encodes_cur_frame
        # If a face was scanned
        if len(encodes_cur_frame) > 0:
            did_scan = True

        # If a face wasn't found
        else:
            print("No face detected. Retrying...")
            counter += 1

    cap.release()  # Release the video capture object
    if counter == 100:
        raise ValueError("No face found.")

    return img_rgb, encodes_cur_frame  # Return the image's rgb and encoding


'''
The purpose of this function is to store important information in the lists
passed in as parameter to this function. This includes each president's image and
encoding, and each president's name. This will allow these values to be used later
in the program to compare the user's face to the each president's face to find the best
match. This function doesn't return a value since each list will be modified. The 
presidents list is the list of each president stored from the ImagesPresidents directory,
the path is the access to the directory with the images, images is a list that will contain
each president's image and encoding, and names will be a list of each president's name.
'''
def fill_presidents(presidents: List[str], path: str, images: List[Tuple[np.ndarray, np.ndarray]],
                    names: List[str]) -> None:
    if not os.path.exists(path):
        raise ValueError("Invalid Path.")

    full_paths = [os.path.join(path, president) for president in presidents]
    # Loop through each president in the list.
    for president in presidents:
        # Get the current president's image.
        cur_img = cv2.imread(f"{path}/{president}")

        if cur_img is None:
            print(f"Could not read image {president}. Skipping.")
            continue

        try:
            # Get the current images RGB value.
            img_rgb = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        except cv2.error as e:
            print(f"Error converting {president} to RGB:{e}. Skipping.")
            continue

        # Get the current images facial encoding.
        encode = face_recognition.face_encodings(img_rgb)

        # If the encoding found was valid, append the data to the lists.
        if len(encode) > 0:
            images.append((cur_img, encode[0]))
            names.append(os.path.splitext(president)[0])
        # If the encoding wasn't valid
        else:
            print(f"No face detected in {president}. Skipping.")


'''
The purpose of this function is to find the closest possible match from the 
president's list compared to the image scanned in from the user. This will 
involve using the face recognitions compare faces function on the list of president's
to find the best possible match. This matches index location will be one value returned to 
the user, as well as a similarity percentage based on how similar the user is to the president they
matched with. The user parameter is the users scanned image, and the presidents parameter is the list
of all the president's images, each represented using n-dimensional arrays.
'''
def find_match(user: np.ndarray, presidents: List[np.ndarray]) -> Tuple[int, float]:
    # Ensure a correct user was provided.
    if not isinstance(user, np.ndarray) or user.size == 0 or user.shape != (128,):
        raise ValueError("Invalid user encoding.")

    # Ensure the presidents provided is valid.
    valid_presidents = [p for p in presidents if p.size > 0]
    # Raise an exception if no presidents were verified.
    if len(valid_presidents) == 0:
        raise ValueError("No valid presidents.")

    # Find the best possible match.
    match = face_recognition.compare_faces(valid_presidents, user)
    face_dist = face_recognition.face_distance(valid_presidents, user)

    # If no match was found.
    if len(face_dist) == 0:
        raise ValueError("No distances found.")

    # Get the match index and the similarity percentage.
    match_index = np.argmin(face_dist)
    best_match_dist = face_dist[match_index]
    similarity = (1 - best_match_dist) * 100

    return match_index, similarity  # Return the match index and similarity
