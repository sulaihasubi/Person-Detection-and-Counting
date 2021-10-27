### Python Developer: Sulaiha Subi ###
### Version 1.3: 22nd October 2021###
### Python Version: 3.9 ###
### Face Recognition in Video to detect people in Designated Area ###

# Library 1st Model
import cv2
import imutils
import time
import mediapipe as mp
import numpy as np
import face_recognition
import os
from datetime import datetime
import numpy as np  # for mathematical operations


# path = '/Users/risehill/Documents/15-Aerodyne/FaceDetectionVideo/faceDatabase'
path = 'faceDatabase'
images = []
classNames = []
count = 0  # this is for counting frame from the video read
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')
# cap = cv2.VideoCapture("VideoEnterNonProductive.mp4")
cap = cv2.VideoCapture("videos/VideoEnterNonProductive.mp4")

# For FPS purpose
fps_start_time = datetime.now()
fps = 0
total_frames = 0

# Count the number of frames
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rate = cap.get(cv2.CAP_PROP_FPS)


# Print details
print("Total Frames:", length)
print("Frames Width:", width)
print("Frames Height:", height)
print("Frames Rate:", rate)

# Start calculate the duration of recognised person
# fps_start_time = datetime.now()
fps = 0

# Creating dictionary to count duration
object_id_list = []
dtime = dict()
dwell_time = dict()
dwell_time2 = dict()
total_frames = 0

# Export the video results
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# Define output of the video (This is version 2): Define Video Writer Objects
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# out = cv2.VideoWriter('/Users/risehill/Documents/15-Aerodyne/FaceDetectionVideo/Enter-Non-Productive.mp4', fourcc, 60,
#                       (frame_width, frame_height), isColor=True)


# Define output: Crete Images frame by frame, start with frame0. Read from the raw video input
# Output path is Images
# Uncomment this to convert frame into Images
# path2 = '/Users/risehill/Documents/15-Aerodyne/FaceDetectionVideo/Images'
# i = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite(os.path.join(path2, '' + str(i) + '.jpg'), frame)
#     i += 1
# print("Done!")

# This is for face detection purposes only.
# mpFaceDetection = mp.solutions.face_detection
# mpDraw = mp.solutions.drawing_utils
# faceDetection = mpFaceDetection.FaceDetection(0.75)
#
# save_json = True
# content_json = []
# i = 0
# pTime = 0

# For frame counter and its timestamp
currentFrame = 0
frame_counter = 0
outputFrameIndices = []

# For testing
faceFound = []
ts = [0,0,False]


while True:
    success, img = cap.read()

    # Extract the frame
    ret, frame = cap.read()  # read current frame
    milisec = cap.get(cv2.CAP_PROP_POS_MSEC)

    # FPS Value
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime

    # resizing the frame
    # frame = imutils.resize(frame, width=600)
    # total_frames = total_frames + 1

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # This is for Face Detection purposes
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # results = faceDetection.process(imgRGB)
    # if results.detections:
    #     for id, detection in enumerate(results.detections):
    #         mpDraw.draw_detection(img, detection)
    #         print(id, detection)
    #         print(detection.score)
    #         print(detection.location_data.relative_bounding_box)


    person = 1
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        # print(matches)
        matchIndex = np.argmin(faceDis)
        # print(matchIndex)

        # This is for FPS if needed but not that accurate
        # fps_end_time = datetime.now()
        # time_diff = fps_end_time - fps_start_time
        # if time_diff.seconds == 0:
        #     fps = 0.0
        # else:
        #     fps = (total_frames / time_diff.seconds)
        #
        # fps_text = "FPS: {:.2f}".format(fps)

        if matches[matchIndex]:
            # name = classNames[matchIndex].upper()+" "+str(i)
            name = classNames[matchIndex].upper()
            # count = len([img.index(i) for i in matches])
            # print("The Match indices list count is : " + str(count))
            person += 1
            # print(person)

            # Count the first and last frame of the person recognised in the video
            # Store Frame in Arrays
            # Capture frame-by-frame and count the its timestamp
            frame_counter = frame_counter + 1
            outputFrameIndices.append(frame_counter)
            frame_timestamp = milisec
            if frame is None:
                break
            print(name, frame_counter, frame_timestamp)

            # Model for calculation: Calculate duration of detected faces (people)
            if name not in object_id_list:
                object_id_list.append(name)
                dtime[name] = datetime.now()
                dwell_time[name] = 1
                dwell_time2[name] = 1

            else:
                curr_time = datetime.now()
                old_time = dtime[name]
                time_diff = curr_time - old_time
                dtime[name] = datetime.now()
                sec = time_diff.total_seconds()
                # secs = now.strftime('%H:%M:%S')
                mins = sec / 60

                dwell_time[name] += mins
                dwell_time2[name] += sec
                # print(dwell_time)
                # print(curr_time)

            # To show the timestamp
            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            duration = "{} | {}sec".format(name, int(dwell_time2[name]))
            # duration = "{} | {}min:{}sec".format(name, int(dwell_time[name]), int(dwell_time2[name]))
            # print(duration)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, duration, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Number of People in Non-Productive(PT):{person - 1}', (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
            # cv2.putText(img, f'Timestamp:{timestamp}', (10, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
            # cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow('Video', img)
    # out.write(img)
    # cv2.waitKey(1)s

    # Press S to break the Window
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# When everything done, release the video capture and video write objects
cap.release()
# out.release()
# result.release()

# Closes all the frames
cv2.destroyAllWindows()
