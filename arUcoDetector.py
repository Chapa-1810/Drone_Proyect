import cv2
import numpy as np
from djitellopy import Tello
import time

DEBUG = False

tello = Tello()

Hz = 2

if not DEBUG:
    print("Connecting to Tello")
    tello.connect()

    tello.streamon()
    frame_reader = tello.get_frame_read()

    print("Taking off")
    tello.takeoff()


# each tuple for x, y contains the following:
# percentage distance from center to edge of frame
# sign of the distance
kP =0.3 #0.3 # Proportional control constant
last_detections = []
yaw_constant_vel = 20
def controller(x_percentage , y_percentage , z_distance, aruco_yaw, detection):
    print(x_percentage , y_percentage , z_distance, aruco_yaw, detection)
    if DEBUG:
        return 
    
    # If no detection, hover
    if not detection and len(last_detections) > 0:
        last_sign_x = last_detections[-1][0] / abs(last_detections[-1][0]) if last_detections[-1][0] != 0 else yaw_constant_vel
        tello.send_rc_control(0, 0, 0, int(yaw_constant_vel * last_sign_x))
        return
    
    x_proportional = kP * x_percentage if abs(x_percentage) < 60 else 0
    y_proportional = kP * y_percentage
    z_proportional = (kP * 0.8) * z_distance    
    yaw_proportional = kP * x_percentage if abs(x_percentage) >= 60 else 0

    yaw_proportional = min(yaw_proportional, yaw_constant_vel * 2 )

    tello.send_rc_control(int(x_proportional), int(z_proportional), int(y_proportional) * -1, int(yaw_proportional))

    last_detections.append((int(x_proportional), int(z_proportional), int(y_proportional), int(yaw_proportional)))

def get_rotation_from_corners(corners):
    # Define the real-world coordinates of the marker corners
    marker_size = 0.06  # The actual size of the marker (in meters or any unit)
    obj_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, marker_size / 2, 0],
        [marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    # Convert corners to the correct format
    img_points = np.array(corners, dtype=np.float32).reshape((4, 2))

    # Assume a simple pinhole camera model with fx = fy = 1 and cx = cy = 0
    camera_matrix = np.array([
        [ 908.2856024835826 ,  0.0 ,  482.87858112836204 ], 
        [ 0.0 ,  901.8162522058758 ,  333.0625329073474 ], 
        [ 0.0 ,  0.0 ,  1.0 ]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    # Use solvePnP to get the rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

    if not success:
        raise Exception("Could not solve PnP problem")

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return rotation_matrix, tvec

aruco_dicts = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}

aruco_dict = aruco_dicts["DICT_4X4_50"]
aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dictionary,aruco_params)

center = None
cap = None

target_distance = 0.5

if not DEBUG:
    frame = frame_reader.frame
else: 
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

distances_x = []
distances_y = []
distances_z = []
rotatation_yaw = []
percentage_x, percentage_y, avg_distance_z, avg_rotation_yaw = 0, 0, 0, 0

while True:
    if not DEBUG:
        frame = frame_reader.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    else:
        _, frame = cap.read()
    center = (frame.shape[1] // 2, frame.shape[0] // 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    detected = ids is not None
    if detected:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Get centroid of aruco marker
        for i in range(len(ids)):
            rotation_matrix, tvec = get_rotation_from_corners(corners[i])

            # Get euler angles from rotation matrix
            angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        
            # Get distance from camera to marker using only z value
            distance = tvec[2]

            c = corners[i][0]
            x = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4
            y = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.line(frame, (int(x), int(y)), center, (0, 0, 255), 2)

            # Append distances to list for averaging
            distances_x.append(x - center[0])
            distances_y.append(y - center[1])
            distances_z.append(distance)
            rotatation_yaw.append(angles[2])

        
        # Calculate average distance_x
        avg_distance_x = np.mean(distances_x)
        distances_x = []

        # Avergae distance but in percetnage from the center to the edge of the frame
        percentage_x = avg_distance_x / (frame.shape[1] // 2) * 100
        
        # Calculate average distance_y
        avg_distance_y = np.mean(distances_y)
        distances_y = []

        # Calculate average distance_z
        avg_distance_z = np.mean(distances_z)
        distances_z = []

        # Get percentage distance_z from the target distance
        avg_distance_z = (avg_distance_z - target_distance) / target_distance * 100

        # Avergae distance but in percetnage from the center to the edge of the frame
        percentage_y = avg_distance_y / (frame.shape[0] // 2) * 100

        # Calculate average rotation_yaw
        avg_rotation_yaw = np.mean(rotatation_yaw)
        rotatation_yaw = []



        # Draw distance on frame
        cv2.putText(frame, f"{percentage_x:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw distance on frame
        cv2.putText(frame, f"{percentage_y:.2f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw distance on frame
        cv2.putText(frame, f"{avg_distance_z:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw distance on frame
        cv2.putText(frame, f"{avg_rotation_yaw:.2f} rad", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    controller(percentage_x, percentage_y, avg_distance_z, avg_rotation_yaw, detected)
    
    cv2.imshow("frame", frame)
    
    #time.sleep(1/Hz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if DEBUG:
    cap.release()
else:
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

    print("Landing")
    tello.land()