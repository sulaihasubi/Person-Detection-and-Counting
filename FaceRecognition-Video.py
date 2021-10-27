### Python Developer: Sulaiha Subi ###
### Version 1.1: 21st October 2021###
### Python Version: 3.9 ###
### Face Recognition in Video to detect people in Designated Area ###

import cv2
import face_recognition
import img as img
import numpy as np

input_movie = cv2.VideoCapture("sample_video2.mp4")
# output_movie = cv2.VideoWriter
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

image = face_recognition.load_image_file("sample_image.jpeg")
face_encoding = face_recognition.face_encodings(image)[0]

known_faces = [
face_encoding,
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Phani Srikant"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        print("Writing frame {} / {}".format(frame_number, length))


    # Display
    # show the frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Output", ret)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))
        fps = int(input_movie.get(cv2.CAP_PROP_FPS))
        frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_movie = cv2.VideoWriter("output.mp4", codec, fps, (frame_width, frame_height))
        output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
output_movie.release()
frame.release()
ret.release()