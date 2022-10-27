from flask import Flask, render_template, Response
import numpy as np
import cv2
import mediapipe as medPipe

app = Flask(__name__)

# calculate angle of each position between joints


def calcAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rads = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rads*180.0/np.pi)

    if ang > 180.0:
        ang = 360-ang

    return ang


def gen_curls():
    drawings = medPipe.solutions.drawing_utils
    poseSolutions = medPipe.solutions.pose

# INIT VARIABLE FOR REPS AND MOTION
    reps = 0
    motion = None

    cap = cv2.VideoCapture(0)
    # creating mediapipe instance with a 70 percent confidence accuracy
    with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
        while cap.isOpened():
            ret, frame = cap.read()

            # change image to rgb to allow mediapipe to process images
            imageColoring = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageColoring.flags.writeable = False

            # detect the image for joints
            detections = poseTrack.process(imageColoring)

            # change image back to brg to allow openCV to process image
            imageColoring.flags.writeable = True
            imageColoring = cv2.cvtColor(imageColoring, cv2.COLOR_RGB2BGR)

            # Get the landmarks of the shoulders, elbows, and wrist to calculate the angle
            try:
                bodyPositions = detections.pose_landmarks.landmark

                # GET LEFT ARM COORDINATES
                L_shoulder = [bodyPositions[poseSolutions.PoseLandmark.LEFT_SHOULDER.value].x,
                              bodyPositions[poseSolutions.PoseLandmark.LEFT_SHOULDER.value].y]
                L_elbow = [bodyPositions[poseSolutions.PoseLandmark.LEFT_ELBOW.value].x,
                           bodyPositions[poseSolutions.PoseLandmark.LEFT_ELBOW.value].y]
                L_wrist = [bodyPositions[poseSolutions.PoseLandmark.LEFT_WRIST.value].x,
                           bodyPositions[poseSolutions.PoseLandmark.LEFT_WRIST.value].y]

                # GET RIGHT ARM COORDINATES
                R_shoulder = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_SHOULDER.value].x,
                              bodyPositions[poseSolutions.PoseLandmark.RIGHT_SHOULDER.value].y]
                R_elbow = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_ELBOW.value].x,
                           bodyPositions[poseSolutions.PoseLandmark.RIGHT_ELBOW.value].y]
                R_wrist = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_WRIST.value].x,
                           bodyPositions[poseSolutions.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle of each arm
                leftArmAngle = calcAngle(L_shoulder, L_elbow, L_wrist)
                RightArmAngle = calcAngle(R_shoulder, R_elbow, R_wrist)

                # if both arms are down keep the motion down
                if leftArmAngle > 160 and RightArmAngle > 160:
                    motion = "  down"
                # if left arm is curling and right arm is down increase reps and change motion
                if leftArmAngle < 30 and RightArmAngle > 160 and motion == '  down':
                    motion = "  up"
                    reps += 1
                # if right arm is curling and left arm is down increase reps and change motion
                if RightArmAngle < 30 and leftArmAngle > 160 and motion == '  down':
                    motion = "  up"
                    reps += 1
                # if both arms are curling then increase rep and change motion
                if leftArmAngle < 30 and RightArmAngle < 30 and motion == "  down":
                    motion = "  up"
                    reps += 1

            except:
                pass

            # VARS FOR TEXT FIELD
            image = imageColoring
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 2.5
            color = (255, 255, 0)
            thickness = 2

            # COUNTER TEXT
            cv2.putText(image, str(reps), (10, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, motion, (60, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)

            # Render the landmarks and dictate their color and thickens
            drawings.draw_landmarks(imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS, drawings.DrawingSpec(
                color=(255, 0, 255), thickness=2, circle_radius=2), drawings.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

            cv2.imshow('Image Feed', imageColoring)
            # Exit by pressing 'X'
            if cv2.waitKey(10) & 0xFF == ord('x'):
                print(x)
                break
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')
        cap.release()
        cv2.destroyAllWindows()


def gen_pushups():
    # two libraries from mediapipe to recognize poses
    drawings = medPipe.solutions.drawing_utils
    poseSolutions = medPipe.solutions.pose

    # CAPTURE THE VIDEO
    cap = cv2.VideoCapture(0)

    # INIT VARIABLE FOR REPS AND MOTION
    reps = 0
    motion = None

    # creating mediapipe instance with a 70 percent confidence accuracy
    with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
        while cap.isOpened():
            ret, frame = cap.read()

            # change image to rgb to allow mediapipe to process images
            imageColoring = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageColoring.flags.writeable = False

            # detect the image for joints
            detections = poseTrack.process(imageColoring)

            # change image back to brg to allow openCV to process image
            imageColoring.flags.writeable = True
            imageColoring = cv2.cvtColor(imageColoring, cv2.COLOR_RGB2BGR)

            # Get the landmarks of all body positions
            try:
                bodyPositions = detections.pose_landmarks

                # store all coordinates of all the landmark poses
                landmarkList = []

                # if body is present in camera
                if bodyPositions:
                    drawings.draw_landmarks(
                        imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS)
                    # iterate thru all of the landmarks to store and compare later
                    for id, landmark in enumerate(detections.pose_landmarks.landmark):
                        print(id)
                        height, width, _ = imageColoring.shape
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        landmarkList.append([id, x, y])
                # check if body was present
                if len(landmarkList) != 0:
                    # logic to check if joint positions of left/right shoulder and left/right elbow change
                    if ((landmarkList[12][2] - landmarkList[14][2]) >= 15 and
                            (landmarkList[11][2] - landmarkList[13][2]) >= 15):
                        motion = "  down"
                    if ((landmarkList[12][2] - landmarkList[14][2]) <= 5 and
                            (landmarkList[11][2] - landmarkList[13][2]) <= 5) and motion == "  down":
                        motion = "  up"
                        reps += 1
                        print(reps)

            except:
                pass

            # VARS FOR TEXT FIELD
            image = imageColoring
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 2.5
            color = (255, 255, 0)
            thickness = 2

            # COUNTER TEXT
            cv2.putText(image, str(reps), (10, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, motion, (60, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)

            # Render the landmarks and dictate their color and thickens
            drawings.draw_landmarks(imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS, drawings.DrawingSpec(
                color=(255, 0, 255), thickness=2, circle_radius=2), drawings.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

            cv2.imshow('Image Feed', imageColoring)

            # Exit by pressing 'X'
            if cv2.waitKey(10) & 0xFF == ord('x'):
                break
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')
        cap.release()
        cv2.destroyAllWindows()


def gen_squats():
    # two libraries from mediapipe to recognize poses
    drawings = medPipe.solutions.drawing_utils
    poseSolutions = medPipe.solutions.pose

    # CAPTURE THE VIDEO
    cap = cv2.VideoCapture(0)

    reps = 0
    motion = None

    # creating mediapipe instance with a 70 percent confidence accuracy
    with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
        while cap.isOpened():
            ret, frame = cap.read()

            # change image to rgb to allow mediapipe to process images
            imageColoring = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageColoring.flags.writeable = False

            # detect the image for joints
            detections = poseTrack.process(imageColoring)

            # change image back to brg to allow openCV to process image
            imageColoring.flags.writeable = True
            imageColoring = cv2.cvtColor(imageColoring, cv2.COLOR_RGB2BGR)

            # Get the landmarks of the shoulders,hip, knee, and ankle to calculate the angle
            try:
                bodyPositions = detections.pose_landmarks.landmark

                # GET SQUAT COORDINATES
                # only need one side so we are choosing right landmarks
                R_hip = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_HIP.value].x,
                         bodyPositions[poseSolutions.PoseLandmark.RIGHT_HIP.value].y]
                R_knee = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_KNEE.value].x,
                          bodyPositions[poseSolutions.PoseLandmark.RIGHT_KNEE.value].y]
                R_ankle = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_ANKLE.value].x,
                           bodyPositions[poseSolutions.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate the knee-joint angle and the hip-joint angle
                kneeAngle = calcAngle(R_hip, R_knee, R_ankle)
                kneeAngle = round(kneeAngle, 2)

                # if standing up
                if kneeAngle > 150:
                    motion = "  up"
                # if doing squat motion down
                if kneeAngle <= 120 and motion == "  up":
                    print(kneeAngle)
                    motion = "  down"
                    reps += 1

            except:
                pass

            # VARS FOR TEXT FIELD
            image = imageColoring
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 2.5
            color = (255, 255, 0)
            thickness = 2

            # COUNTER TEXT
            cv2.putText(image, str(reps), (10, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(image, motion, (60, 60), font,
                        fontScale, color, thickness, cv2.LINE_AA)

            # Render the landmarks and dictate their color and thickens
            drawings.draw_landmarks(imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS, drawings.DrawingSpec(
                color=(255, 0, 255), thickness=2, circle_radius=2), drawings.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

            cv2.imshow('Image Feed', imageColoring)
            # Exit by pressing 'X'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame + b'\r\n')


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/curls/')
def curls():
    return Response(gen_curls(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/pushups/")
def pushups():
    return Response(gen_pushups(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/squats/')
def squats():
    return Response(gen_squats(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
