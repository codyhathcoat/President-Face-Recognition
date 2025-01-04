from Presidents import *


def main():
    path = "ImagesPresidents"  # Use the directory with the president's pictures
    images = []  # A list for all the president's images
    president_names = []  # A list of all the president's names
    president_jpg_list = os.listdir(path)  # The list of president jpg's.
    choice = 0  # Holds user input

    # Greet the user and make sure they have a usable web camera.
    print("Welcome to President Face Recognition!\n")
    print("Here we will use your devices camera to scan your face, and compare the results to all US Presidents as of "
          "2024 and find your closet match.")

    # Loop until a choice has been made
    while choice != 1 and choice != 2:
        temp_choice = input("Do you have a web camera, and give permission for it to be used?\n1. Yes\n2. No\n")

        # Make sure the choice was an integer
        try:
            choice = int(temp_choice)

        except ValueError:
            print("Try again with an integer (1 for yes, 2 for no).")

        # If the user wishes to continue
        if choice == 1:
            # Try and scan the users face.
            try:
                img, encodes = get_face()  # Use the webcam to scan a face.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            except ValueError as e:
                print(f"Error when scanning your face: {e} Please try again.")
                exit(1)

            # Try and get the images of the presidents.
            try:
                # Go through each image and get the names, images, and encodings in their lists
                print("Searching through each President to find your closest match...")
                fill_presidents(president_jpg_list, path, images, president_names)  # Get the names and images encoded
                # encode_list_known = find_encodings(images)

            except ValueError as e:
                print(f"Error: {e} Please try again.")
                exit(1)

            # Try and find the best matching president.
            try:
                # Find the best match and how similar they are.
                best_index, similarity = find_match(encodes[0], [encoding for _, encoding in images])

            except ValueError as e:
                print(f"Error: {e} Please try again.")
                exit(1)

            # Tell the user who their best match is and how similar they are.
            print("The president you match best with is", president_names[best_index])
            print("You match with a similarity percentage of:", similarity, "percent")

            president_img, _ = images[best_index]  # Get the correct president image.

            # Get the users face location
            user_face_loc = face_recognition.face_locations(img)[0]
            # Draw a green rectangle around the users face.
            cv2.rectangle(img, (user_face_loc[3], user_face_loc[0]), (user_face_loc[1], user_face_loc[2]),
                          (0, 255, 0, 2), 2)

            # Get the presidents face location
            president_face_loc = face_recognition.face_locations(president_img)[0]
            # Draw a green rectangle around the presidents face.
            cv2.rectangle(president_img, (president_face_loc[3], president_face_loc[0]),
                          (president_face_loc[1], president_face_loc[2]), (0, 255, 0, 2), 2)

            # Put text around the users face to show their match name and similarity percentage.
            cv2.putText(img, f"Match: {president_names[best_index]} ", (user_face_loc[3] - 100, user_face_loc[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(img, f"Similarity: {round(similarity)}%", (user_face_loc[3] - 6, user_face_loc[2] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # Show the user their found image and their matching president's image
            print("Displaying the user image and the presidents image to your screen. These images will display for "
                  "15 seconds, then the program will automatically suspend. Thank you!")

            cv2.imshow("User", img)
            cv2.imshow("President", president_img)
            cv2.waitKey(15000)
            cv2.destroyAllWindows()

        # Otherwise the user can't use the program without their camera.
        elif choice == 2:
            print("Please enable the webcam and try again.")

        else:
            print("Invalid input, try again.")


if __name__ == "__main__":
    main()
