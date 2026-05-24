# =========================================== #
#            Libraries and Dependencies       #
# =========================================== #
import json
import cv2
import numpy as np

import time
import base64
import paho.mqtt.client as mqtt
import sys
import os

sys.stderr = open(os.devnull, 'w')
# =========================================== #
#            Parameters and Assets           #
# =========================================== #

MODEL_PATH = "detect.tflite"
LABELS_PATH = "labelmap.txt"
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

# ------------- Load Labels ------------------
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --------------- Load TensorFlow Model ----------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']
# for usb logitech c270  cap = cv2.VideoCapture(2, cv2.CAP_V4L2) 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
prev_time = 0
debug = None
last_x_int = 0

# Main stream resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)# 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# 720




# Function to order colibrated points from flutter app, because user can point them in any order, 
# we have to find which one is lef/right/top/buttom

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


# ========================== MQTT mosquitto ================================== 
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




# =================== Main Loop =========================== 
# =========================================================

while True:
   
    ret, frame = cap.read()
    if not ret:
        break
    # preparing frame to send to flutter app if needed
    last_frame = frame.copy()
    orig_h, orig_w = frame.shape[:2]


    warped_cx, warped_cy = None, None
    # Отрисовка зон на оригинальном кадре
    for i in range(1, LINES_AMOUNT):
        x_line = int(i * orig_w / LINES_AMOUNT)
        cv2.line(frame, (x_line, 0), (x_line, orig_h), (255, 255, 255), 2)

    # Подготовка кадра под модель
    img = cv2.resize(frame, (input_width, input_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    # Инференс
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    person_count = 0
    

    for i in range(len(scores)):

        if scores[i] > CONF_THRESHOLD and classes[i] == PERSON_CLASS_ID:
            y1, x1, y2, x2 = boxes[i]
            x1 = int(x1 * orig_w) 
            x2 = int(x2 * orig_w)
            y1 = int(y1 * orig_h)
            y2 = int(y2 * orig_h)

            cx = (x1 + x2) // 2 # X coordinate of the centere
            cy = y2  # Y coordinate of the centere buttom

      
            # Отрисовка на оригинальном кадре
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            person_count += 1
            
            # Если калибровка выполнена, трансформируем в warped координаты
            if calibrated and M is not None:
                
                point = np.array([[[cx, cy]]], dtype=np.float32)

            # transform point
                warped_point = cv2.perspectiveTransform(point, M)

                cx_warped = int(warped_point[0][0][0])
                cy_warped = int(warped_point[0][0][1])

                warped_cx = cx_warped
                warped_cy = cy_warped

          
                zone = int(x_norm * NUM_ZONES)
                zone = max(0, min(NUM_ZONES - 1, zone))

                if calibrated and WARP_W > 0:
                    x_norm = cx_warped / WARP_W
                    x_norm = np.clip(x_norm, 0.0, 1.0)
                    

                    y_norm = cy_warped / WARP_H
                    y_norm = np.clip(y_norm, 0.0, 1.0)
                    
                    x_sent = x_norm
                    y_sent = y_norm

                else:
                    continue

                payload = json.dumps({
                    
                    "zone": zone
                })

        

                payload_coords = json.dumps({
                     "type": "player_position",
                        "x": x_sent,
                        "y": y_sent,
                        "ts": time.time()
                    
                })

                                
                if x_sent != last_x_sent or y_sent != last_y_sent:
                    mqtt_client.publish(MQTT_TOPIC_PUB, payload_coords)
                    last_x_sent = x_sent
                    last_y_sent = y_sent

                if zone != last_zone:
                    mqtt_client.publish("servo/control", payload)
                    last_zone = zone

                
                


                

                

                cx_warped_text = f"Warped Zone: {zone}"
                #cx_normalized_text = f"Normalized X: {x_sent}"
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
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
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
