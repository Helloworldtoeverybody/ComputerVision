


# workong tracking yolo + everything + kalman, but slow fps






# =========================================== #
#            Libraries and Dependencies       #
# =========================================== #
import json
import cv2
import numpy as np
import tensorflow as tf
import time
import base64
import paho.mqtt.client as mqtt
import sys
import os
from ultralytics import YOLO

sys.stderr = open(os.devnull, 'w')
# =========================================== #
#            Parameters and Assets           #
# =========================================== #


PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.5 # Порог уверенности
LINES_AMOUNT = 12
NUM_ZONES = 12
zone = 0 
last_zone = 0
last_x_sent = 0
last_y_sent = 0
need_frame = False
last_frame = None
current_zone = 0
x_norm = 0.0
x_sent = 1
calibrated_points = []
WARP_H = 0
WARP_W = 0
calibrated = False
M = None
cx = 0

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "calibration/frame/request"
MQTT_TOPIC_PUB = "calibration/frame/response"
MQTT_TOPIC_PUB_ESP = "calibration/frame/response"




model = YOLO("yolov8n_openvino_model")

# for usb logitech c270  cap = cv2.VideoCapture(2, cv2.CAP_V4L2) 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
prev_time = 0
debug = None
last_x_int = 0


cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)# 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# 720


print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



# =========================================== #
#                   MQTT                       #
# =========================================== #

def order_points_safe(pts):
    pts = np.array(pts, dtype=np.float32)

    # sort by y
    pts = pts[np.argsort(pts[:, 1])]

    top = pts[:2]
    bottom = pts[2:]

    # sort left to right
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def compute_warp_size(pts):
    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    W = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    H = int(max(heightA, heightB))

    return W, H



def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC_SUB)
        print(f"Subscribed to {MQTT_TOPIC_SUB}")
    else:
        print("Failed to connect, return code:", rc)

def on_message(client, userdata, msg):
    global need_frame, calibrated_points, M, calibrated
    global WARP_W, WARP_H

    payload = msg.payload.decode()
    print(f"MQTT received raw: {payload}")

    

    try:
        data = json.loads(payload)  # <-- парсим JSON
        if data.get("msg") == "FRAME_NEEDED":
            need_frame = True
            print("✅ FRAME_NEEDED detected")
        if "points" in data:
            calibrated_points = data["points"]
            print(calibrated_points)
            pts = np.array(
                [[p["x"], p["y"]] for p in calibrated_points],
                dtype=np.float32
            
            )
            src = order_points_safe(pts)
   
            # ---------- DEBUG: visualize calibration quad ----------
            if last_frame is not None:
                global debug
                debug = last_frame.copy()
                cv2.polylines(
                    debug,
                    [src.astype(np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=3
                )

                for (x, y) in src:
                    cv2.circle(
                        debug,
                        (int(x), int(y)),
                        6,
                        (0, 0, 255),
                        -1
                    )

                
            # -------------------------------------------------------
            WARP_W, WARP_H = compute_warp_size(src)
            dst = np.array([
            [0, 0],
            [WARP_W - 1, 0],
            [WARP_W - 1, WARP_H - 1],
            [0, WARP_H - 1]
        ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            calibrated = True
            
            

            
        print("colibrtaed")

    except json.JSONDecodeError:
        print("❌ Payload is not JSON")

# Create MQTT client once
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# =========================================== #
#             Helper Functions                #
# =========================================== #

def detect_zone_by_x(x_center, width):
    third = width // LINES_AMOUNT
    for i in range(LINES_AMOUNT):
        if x_center < third:
            return 0
        elif x_center > i*third and x_center < (i+2)*third:
            return i+1
    return 0

# =========================================== #
#                Main Loop                     #
# =========================================== #
print("ACTUAL:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


































# ===========================================
#              KALMAN FILTER
# ===========================================
kf = cv2.KalmanFilter(4, 2)

# state = [x, y, dx, dy]
kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

kf.statePre = np.zeros((4, 1), np.float32)
kf.statePost = np.zeros((4, 1), np.float32)

kalman_initialized = False














while True:
   
    ret, frame = cap.read()
    if not ret:
        break

    last_frame = frame.copy()
    orig_h, orig_w = frame.shape[:2]


    warped_cx, warped_cy = None, None
    # Отрисовка зон на оригинальном кадре
    for i in range(1, LINES_AMOUNT):
        x_line = int(i * orig_w / LINES_AMOUNT)
        cv2.line(frame, (x_line, 0), (x_line, orig_h), (255, 255, 255), 2)

    results = model(frame, imgsz=640, conf=0.4, verbose=False)

    person_count = 0
    

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx_meas = (x1 + x2) // 2
                cy_meas = y2

                # ===== KALMAN FILTER =====
                measurement = np.array([[np.float32(cx_meas)], [np.float32(cy_meas)]])

                if not kalman_initialized:
                    kf.statePost[:2] = measurement
                    kalman_initialized = True

                kf.correct(measurement)
                prediction = kf.predict()

                cx = int(prediction[0])
                cy = int(prediction[1])

                # =========================

                # draw (use SMOOTHED coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                person_count += 1

                # ===== YOUR EXISTING WARP LOGIC =====
                if calibrated and M is not None:
                    point = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_point = cv2.perspectiveTransform(point, M)

                    cx_warped = int(warped_point[0][0][0])
                    cy_warped = int(warped_point[0][0][1])

                    warped_cx = cx_warped
                    warped_cy = cy_warped

                    if WARP_W > 0:
                        x_norm = np.clip(cx_warped / WARP_W, 0.0, 1.0)
                        y_norm = np.clip(cy_warped / WARP_H, 0.0, 1.0)

                        zone = int(x_norm * NUM_ZONES)
                        zone = max(0, min(NUM_ZONES - 1, zone))

                        payload = json.dumps({
                            "zone": zone
                        })

                        payload_coords = json.dumps({
                            "type": "player_position",
                            "x": float(x_norm),
                            "y": float(y_norm),
                            "ts": time.time()
                        })

                        if x_norm != last_x_sent or y_norm != last_y_sent:
                            mqtt_client.publish(MQTT_TOPIC_PUB, payload_coords)
                            last_x_sent = x_norm
                            last_y_sent = y_norm

                        if zone != last_zone:
                            mqtt_client.publish("servo/control", payload)
                            last_zone = zone
                            

                

     
    #cx_normalized_text = f"Basic X: {cx}"
    #cy_normalized_text = f"Basic Y: {cy}"
    cx_warped_text = f"Warped X: {warped_cx}"
    cy_warped_text = f"Warped Y: {warped_cy}"
    warped_width_text = f"Warped width: {WARP_W}"
    warped_heigh_text = f"Warped heigh: {WARP_H}"

    #cv2.putText(frame, cx_warped_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    #cv2.putText(frame, cx_normalized_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    #cv2.putText(frame, cy_normalized_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cx_warped_text , (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cy_warped_text , (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, warped_width_text , (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, warped_heigh_text , (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                



    # Вычисляем текущую зону на warped корте
    current_zone = 0
    if warped_cx is not None:
        third = WARP_W / LINES_AMOUNT
        for i in range(LINES_AMOUNT):
            if warped_cx < (i+1)*third:
                current_zone = i
                break

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Отображение информации
    status_text = f"People: {person_count}  FPS: {int(fps)}"
    zones_text = f"Current Zone: {current_zone}"
    
    cx_text = f"Normal X: {WARP_W}"

    # Отправка кадра, если запросили
    if need_frame and last_frame is not None:
        need_frame = False
        success, buffer = cv2.imencode(".jpg", last_frame)
        if success:
            jpg_as_text = base64.b64encode(buffer).decode()
            mqtt_client.publish(MQTT_TOPIC_PUB, jpg_as_text)

    # Отображаем warped корт
    if calibrated and M is not None:
        warped = cv2.warpPerspective(last_frame, M, (WARP_W, WARP_H))
        if warped_cx is not None and warped_cy is not None:
            #cv2.circle(warped, (int(warped_cx), int(warped_cy)), 5, (0, 0, 255), -1)
        #cv2.imshow("Warped", warped)
            cv2.imshow("CALIBRATION CHECK", debug)

    cv2.imshow("Court Zones", frame)


    inside = (
    warped_cx is not None and
    0 <= warped_cx < WARP_W 
)
    

    if inside:
        print("PERSON INSIDE PLANE")
    else:
        print("PERSON OUTSIDE PLANE")
    




    # ESC для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()

