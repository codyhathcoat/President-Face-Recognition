import os
import pytest
from Presidents import *
from unittest.mock import patch, MagicMock


# Fixture to represent example data.
@pytest.fixture
def setup_data():
    path = "ImagesTest"
    presidents = os.listdir(path)
    images = []
    president_names = []
    return path, presidents, images, president_names


# Fixture mock a face recognition scan.
@pytest.fixture
def mock_face_recognition():
    with patch("face_recognition.compare_faces") as compare_faces, patch(
            "face_recognition.face_distance") as face_distance:
        yield compare_faces, face_distance


@patch("face_recognition.face_locations")  # Mock the face_locations function from the face_recognition library.
@patch("face_recognition.face_encodings")  # Mock the face_encodings function from the face_recognition library.
@patch("cv2.VideoCapture")  # Mock the VideoCapture class from OpenCV cv2 model
# The test cases for the get_face function
class Test_get_face():

    # Test when the results are successful and a face is found
    def test_get_face_success(self, mock_video_capture, mock_face_encodings, mock_face_locations):
        mock_cap_instance = MagicMock()  # Create a mock instance of VideoCapture
        mock_video_capture.return_value = mock_cap_instance  # Make the mock return this instance
        mock_cap_instance.isOpened.return_value = True  # Simulate the camera being available.

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # A 100x100 image
        mock_cap_instance.read.return_value = (True, dummy_frame)

        # Simulate face detection and encoding

        mock_face_locations.return_value = [(10, 20, 30, 40)]  # Mock location
        mock_face_encodings.return_value = ["encoded_face"]  # Mock encoding

        img_rgb, encodes_cur_frame = get_face()  # Call the function

        # Ensure the results match what is expected.
        assert np.array_equal(img_rgb, cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB))
        assert encodes_cur_frame == ["encoded_face"]

    # Test when no face is detected
    def test_get_face_no_face_detected(self, mock_video_capture, mock_face_encodings, mock_face_locations):
        mock_cap_instance = MagicMock()  # Create a mock instance of a VideoCapture object.
        mock_video_capture.return_value = mock_cap_instance  # Make the return this instance.
        mock_cap_instance.isOpened.return_value = True  # Set the camera to opened.
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a frame of a blank image.
        mock_cap_instance.read.return_value = (True, dummy_frame)  # Successfully read the blank image.

        mock_face_locations.return_value = []  # Have the face_locations return an empty list.

        # Call the function to make sure the no face found exception is raised.
        with pytest.raises(ValueError, match="No face found."):
            get_face()

    # Test when the webcam can't be accessed.
    def test_get_face_webcam_access_failure(self, mock_video_capture, mock_face_encoding, mock_face_locations):
        mock_cap_instance = MagicMock()  # Mock a video capture object to this instance.
        mock_video_capture.return_value = mock_cap_instance  # Have the video capture this instance.
        mock_cap_instance.isOpened.return_value = False  # Have the camera fail to open.

        # Call the get_face function to make the right exception is raised.
        with pytest.raises(ValueError, match="Can't access the webcam."):
            get_face()


# Test the function to get the presidents images and names.
class Test_fill_presidents:

    # Test for when the path is invalid.
    def test_invalid_path(self):
        # Test the right exception is raised for an invalid path.
        with pytest.raises(ValueError, match="Invalid Path"):
            fill_presidents([], "invalid_path", [], [])

    # Test for when no list of presidents was provided
    def test_missing_image(self, setup_data):
        path, no_presidents, images, president_names = setup_data  # Get the initial data.
        no_presidents = ["non_existent.jpg"]  # Update the image list to be empty.

        # Test for a missing image.
        with patch("cv2.imread", return_value=None):
            fill_presidents(no_presidents, path, images, president_names)

        # Make sure the list of images and names are still empty
        assert len(images) == 0
        assert len(president_names) == 0

    # Test for when the results should be successful.
    def test_valid_images(self, setup_data):
        # Get the set-up data and call the function.
        path, presidents, images, president_names = setup_data
        fill_presidents(presidents, path, images, president_names)

        # Make sure the presidents are in the list and the flash drive image is not.
        assert "George Washington" in president_names
        assert "John Adams" in president_names
        assert "Thomas Jefferson" in president_names
        assert "flash drive" not in president_names

        assert len(images) == 3  # Make sure there are only 3 images in the list.

        # Make sure each image is valid.
        for img, encoding in images:
            assert img is not None
            assert len(encoding) > 0


# Test the find_match function.
class Test_find_match():

    # Test for when all data is valid.
    def test_find_match_valid(self, mock_face_recognition):
        compare_faces, face_distance = mock_face_recognition  # Get the mock face recognition.

        # Set mock return values for compare_faces and face_distance.
        compare_faces.return_value = [True, False, False]  # First president matches
        face_distance.return_value = [0.2, 0.5, 0.8]  # Example face distance

        # Set up an example user and president encoding.
        user = np.random.rand(128)
        presidents = [np.random.rand(128) for _ in range(3)]
        '''
        presidents = [
            np.random.rand(128),
            np.random.rand(128),
            np.random.rand(128),
        ]
        '''
        # Call the function to find the match index and similarity.
        match_index, similarity = find_match(user, presidents)

        assert match_index == 0  # The first president should be the match
        assert similarity == 80.0  # Similarity should be (1 - 0.2) * 100

    # Test when the find match function isn't given any presidents to compare to.
    def test_find_match_empty_presidents(self, mock_face_recognition):
        # Make sure the function raises the right exception when provided no presidents.
        with pytest.raises(ValueError, match="No valid presidents."):
            match_index, similarity = find_match(np.random.rand(128), [])

    # Test when the find match function isn't given a user to compare to the presidents.
    def test_find_match_invalid_user(self, mock_face_recognition):
        # invalid_user = [0.1, 0.2, 0.3]
        # Set up an invalid user and valid presidents encoding.
        invalid_user = []
        presidents = [np.random.rand(128) for _ in range(3)]

        # Test that the right exception is raised when the user is invalid.
        with pytest.raises(ValueError, match="Invalid user encoding."):
            find_match(invalid_user, presidents)

    # Test when the find_match function can't compute a distance.
    def test_find_match_no_distance(self, mock_face_recognition):
        compare_faces, face_distance = mock_face_recognition  # Get the mock face recognition.

        # Have the compare faces and face distance functions fail.
        compare_faces.return_value = []
        face_distance.return_value = []

        # Have a valid user and presidents list encoding.
        user = np.random.rand(128)
        presidents = [np.random.rand(128) for _ in range(3)]

        # Tes that the right exception is raised when the distance can't be found.
        with pytest.raises(ValueError, match="No distances found."):
            find_match(user, presidents)
