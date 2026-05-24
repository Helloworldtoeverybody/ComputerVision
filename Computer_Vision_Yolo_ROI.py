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
import threading
import math
from scipy.optimize import bisect



# =========================================== #
#            Parameters and Assets           #
# =========================================== #


PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.4 # Порог уверенности
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
custom_drill = []
WARP_H = 0
WARP_W = 0
calibrated = False
M = None
cx = 0
cy = 0

# Реальные размеры половины корта (напротив пушки)
COURT_HALF_LENGTH = 23.77 / 2   # 11.885 м — от сетки до задней линии
COURT_FULL_LENGTH = 23.77       # полная длина корта
COURT_WIDTH       = 8.23        # ширина корта

# Вычисляется при калибровке из 4 точек — не хардкод!
Y_net_computed = COURT_HALF_LENGTH  # fallback до калибровки

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "calibration/frame/request"
MQTT_TOPIC_PUB = "calibration/frame/response"
MQTT_TOPIC_PUB_ESP = "calibration/frame/response"





'''
def calculate_launcher_shot_normalized(X_norm, Y_norm, h_launcher=0.5, wheel_radius=0.05,
                                       h_net=0.914, Y_net=11.885,
                                       court_length=23.77, court_width=8.23):
    # --- Input validation ---
    if X_norm is None or Y_norm is None:
        return None
    
    try:
        X_norm = float(X_norm)
        Y_norm = float(Y_norm)
    except (TypeError, ValueError):
        return None

    if not (0.0 <= X_norm <= 1.0) or not (0.0 <= Y_norm <= 1.0):
        return None  # Out of court bounds

    if Y_norm < 0.01:
        return None  # Too close to launcher, physics breaks down

    # --- Rest of your existing code unchanged ---
    X_p = (X_norm - 0.5) * court_width
    Y_p = Y_norm * court_length
    g = 9.81

    phi = math.atan2(X_p, Y_p)
    R = math.hypot(X_p, Y_p)

    theta_guess = math.radians(15)

    try:
        v_guess = math.sqrt(g * R**2 / (2 * math.cos(theta_guess)**2 * (R * math.tan(theta_guess) + h_launcher)))
    except (ValueError, ZeroDivisionError):
        return None

    def net_func(theta):
        t_net = Y_net / (math.cos(theta) * math.cos(phi) * v_guess)
        z_net = h_launcher + v_guess * math.sin(theta) * t_net - 0.5 * g * t_net**2
        return z_net - h_net - 0.05

    theta_min = math.radians(5)
    theta_max = math.radians(60)

    try:
        theta = bisect(net_func, theta_min, theta_max)
    except ValueError:
        try:
            v_guess *= 1.1
            theta = bisect(net_func, theta_min, theta_max)
        except ValueError:
            return None  # Physics genuinely unsolvable for this position

    try:
        v_final = math.sqrt(g * R**2 / (2 * math.cos(theta)**2 * (R * math.tan(theta) + h_launcher)))
    except (ValueError, ZeroDivisionError):
        return None

    omega = v_final / wheel_radius
    wheel_rpm = omega * 60 / (2 * math.pi)

    return theta, phi, wheel_rpm


'''

def calculate_launcher_shot_normalized(X_norm, Y_norm,
                                       h_launcher=0.5,
                                       wheel_radius=0.05,
                                       h_net=0.914,
                                       Y_net=11.885,
                                       court_length=23.77,
                                       court_width=8.23):
    # Валидация
    if X_norm is None or Y_norm is None:
        return None
    try:
        X_norm, Y_norm = float(X_norm), float(Y_norm)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= X_norm <= 1.0) or not (0.0 <= Y_norm <= 1.0):
        return None
    if Y_norm < 0.01:
        return None

    X_p = (X_norm - 0.5) * court_width
    Y_p = Y_norm * court_length
    g = 9.81

    phi = math.atan2(X_p, Y_p)
    R = math.hypot(X_p, Y_p)

    def net_constraint(theta):
        cos_t = math.cos(theta)
        tan_t = math.tan(theta)
        denom = 2 * cos_t**2 * (R * tan_t + h_launcher)
        if denom <= 0:
            return -999.0
        v = math.sqrt(g * R**2 / denom)
        t_net = Y_net / (v * cos_t * math.cos(phi))
        z_net = h_launcher + v * math.sin(theta) * t_net - 0.5 * g * t_net**2
        return z_net - h_net - 0.05

    theta_min = math.radians(5)
    theta_max = math.radians(60)

    try:
        # Проверяем, что bisect имеет смысл (разные знаки на концах)
        if net_constraint(theta_min) * net_constraint(theta_max) > 0:
            return None
        theta = bisect(net_constraint, theta_min, theta_max)
    except ValueError:
        return None

    cos_t = math.cos(theta)
    denom = 2 * cos_t**2 * (R * math.tan(theta) + h_launcher)
    if denom <= 0:
        return None
    v_final = math.sqrt(g * R**2 / denom)

    omega = v_final / wheel_radius
    wheel_rpm = omega * 60 / (2 * math.pi)

    return theta, phi, wheel_rpm


# model.export(format="openvino")  # run once, creates OpenVINO model

# model = YOLO("yolov8n_openvino_model_first")
model = YOLO("yolov8n_openvino_model")

# for usb logitech c270  cap = cv2.VideoCapture(2, cv2.CAP_V4L2) 
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("❌ Camera failed to open")
    sys.exit()
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
prev_time = 0
debug = None
last_x_int = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)# 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# 720


print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




def calculate_custom_drill():
    payloads = []

    for i in balls:
        x = round(float(i["x"]), 2)
        y = round(float(1.0 - i["y"]), 2)
        delay = i["delay"]

        result = calculate_launcher_shot_normalized(x, y, Y_net=Y_net_computed)
        if result is None:
            print(f"Ball id={i['id']} skipped — unreachable at x={x}, y={y}")
            continue

        theta, phi, rpm = result

        print(f"Ball id={i['id']}")
        print(f"  Elevation θ: {math.degrees(theta):.2f}°")
        print(f"  Horizontal φ: {math.degrees(phi):.2f}°")
        print(f"  Wheel RPM: {rpm:.0f}")
        print(f"  Delay: {delay}")

        payloads.append(json.dumps({
            "type": "custom_drill",
            "id": i['id'],
            "elevation": round(math.degrees(theta), 1),
            "horizontal": round(math.degrees(phi), 1),
            "rpm": int(rpm),
            "delay": delay
        }))

    # Send all shots in one message, newline-separated
    if payloads:
        mqtt_client.publish(MQTT_TOPIC_PUB, "\n".join(payloads))
      
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



def compute_y_net_from_calibration(src_pixels, frame_w, frame_h):
    """
    Вычисляет реальное расстояние от пушки до сетки в метрах.

    src_pixels — 4 угловые точки рабочей половины корта в пикселях,
    упорядоченные как: [top-left, top-right, bottom-right, bottom-left]
    (результат order_points_safe).

    Логика:
      - Игрок отмечает углы РАБОЧЕЙ половины (напротив пушки).
      - Ближний к камере край src → линия сетки.
      - Дальний край src → задняя линия противника.
      - Мировые координаты известны из размеров корта.
      - Обратная гомография даёт мировые координаты пикселя пушки,
        из чего мы находим расстояние до сетки.
    """
    tl, tr, br, bl = src_pixels  # порядок из order_points_safe

    # Мировые координаты 4 углов рабочей половины.
    # Система отсчёта: Y=0 — линия сетки, Y=COURT_HALF_LENGTH — задняя линия.
    # X=0 — центр, X=±COURT_WIDTH/2 — боковые линии.
    half_w = COURT_WIDTH / 2
    world_pts = np.array([
        [-half_w, 0                 ],   # tl — сетка, левый угол
        [ half_w, 0                 ],   # tr — сетка, правый угол
        [ half_w, COURT_HALF_LENGTH ],   # br — задняя линия, правый угол
        [-half_w, COURT_HALF_LENGTH ],   # bl — задняя линия, левый угол
    ], dtype=np.float32)

    pixel_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # Гомография: пиксель → мировые координаты (только рабочая половина)
    H_inv, _ = cv2.findHomography(pixel_pts, world_pts)
    if H_inv is None:
        print("⚠️  compute_y_net: findHomography вернул None, используем fallback")
        return COURT_HALF_LENGTH

    # Пиксельные координаты пушки = центр нижнего края кадра
    # (камера на пушке, смотрит вперёд)
    launcher_px = np.array([[[frame_w / 2.0, frame_h]]], dtype=np.float32)
    launcher_world = cv2.perspectiveTransform(launcher_px, H_inv)[0][0]

    # launcher_world[1] — Y пушки в системе "от сетки".
    # Отрицательное значение = пушка стоит ЗА своей стороной (нормально).
    y_launcher_world = float(launcher_world[1])

    # Расстояние от пушки до сетки:
    # сетка находится при Y=0, пушка при Y=y_launcher_world (<0)
    y_net = abs(y_launcher_world)

    print(f"📐 Пушка в мировых координатах: Y = {y_launcher_world:.3f} м от сетки")
    print(f"📐 Вычисленное расстояние до сетки: Y_net = {y_net:.3f} м")
    return y_net


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC_SUB)
        print(f"Subscribed to {MQTT_TOPIC_SUB}")
    else:
        print("Failed to connect, return code:", rc)

def on_message(client, userdata, msg):
    global balls, need_frame, calibrated_points, M, calibrated
    global WARP_W, WARP_H, Y_net_computed

    payload = msg.payload.decode()
    print(f"MQTT received raw: {payload}")

    

    try:
        data = json.loads(payload)  # <-- парсим JSON
        if data.get("msg") == "FRAME_NEEDED":
            need_frame = True
            print("✅ FRAME_NEEDED detected")

        if data.get("msg") == "custom_start":
            print("✅ Custom drill start")
            calculate_custom_drill()

        if data.get("type") == "custom_drill":

            balls = data["balls"]
            print(f"Recived {len(balls)} balls")

            for ball in balls:
                id = ball["id"]
                x = ball["x"]
                y = ball["y"]

                speed = ball["speed"]
                spin = ball["spin"]
                height= ball["height"]
                delay = ball["delay"]
           
            print(f"BALLS{balls}")
            
 
    

      
                

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

            # Вычисляем реальное расстояние до сетки из калибровочных точек
            if last_frame is not None:
                frame_h, frame_w = last_frame.shape[:2]
                Y_net_computed = compute_y_net_from_calibration(src, frame_w, frame_h)
            else:
                Y_net_computed = COURT_HALF_LENGTH
                print(f"⚠️  last_frame недоступен, Y_net = fallback {Y_net_computed:.3f} м")
            
            

            
       

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



# ---------- Dynamic ROI settings ----------
USE_DYNAMIC_ROI = True

ROI_SIZE = 280        # was 400
MAX_ROI = 450         # was 700
MIN_ROI = 180         # was 250

# And in inference:


FULL_DETECT_EVERY = 6  # 3, 12
frame_count = 0

lost_frames = 0 
MAX_LOST = 5 # 8 

tracking_ok = False




while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    last_frame = frame.copy()

    orig_h, orig_w = frame.shape[:2]
    frame_count += 1
    person_count = 0
    warped_cx, warped_cy = None, None


    run_full_frame = (
        not tracking_ok or
        lost_frames > MAX_LOST or
        frame_count % FULL_DETECT_EVERY == 0
    )

    # ----------------------------
    # DYNAMIC ROI CROP
    # ----------------------------
    if USE_DYNAMIC_ROI and not run_full_frame and kalman_initialized:
        pred = kf.predict()
        pred_x, pred_y = int(pred[0,0]), int(pred[1,0])
        vx, vy = abs(float(pred[2,0])), abs(float(pred[3,0]))
        
        roi_size = int(np.clip(400 + (vx + vy) * 20, MIN_ROI, MAX_ROI))

        x1_roi = max(0, pred_x - roi_size // 2)
        y1_roi = max(0, pred_y - roi_size // 2)
        x2_roi = min(orig_w, pred_x + roi_size // 2)
        y2_roi = min(orig_h, pred_y + roi_size // 2)

        roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

        if roi.size == 0 or x2_roi <= x1_roi or y2_roi <= y1_roi:
            tracking_ok = False
            lost_frames += 1
            continue

        results = model(roi, imgsz=320, conf=0.4, verbose=False)  # was 448
        roi_offset_x, roi_offset_y = x1_roi, y1_roi
    else:
        results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        roi_offset_x, roi_offset_y = 0, 0

    # Variable to track if a person was actually found in this frame
    detection_in_frame = False

    for r in results:
        if len(r.boxes) > 0:
            detection_in_frame = True

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > CONF_THRESHOLD:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 += roi_offset_x
                x2 += roi_offset_x
                y1 += roi_offset_y
                y2 += roi_offset_y

                cx_meas = (x1 + x2) // 2
                cy_meas = y2

                measurement = np.array([[np.float32(cx_meas)], [np.float32(cy_meas)]])

                if not kalman_initialized:
                    kf.statePost[:2] = measurement
                    kalman_initialized = True

                kf.correct(measurement)
                cx, cy = int(kf.statePost[0,0]), int(kf.statePost[1,0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # ===== WARP LOGIC =====
                if calibrated and M is not None:
                    point = np.array([[[cx, cy]]], dtype=np.float32)
                    warped_point = cv2.perspectiveTransform(point, M)

                    cx_warped = int(warped_point[0][0][0])
                    cy_warped = int(warped_point[0][0][1])
                    warped_cx, warped_cy = cx_warped, cy_warped

                    if WARP_W > 0:
                        x_norm = np.clip(cx_warped / WARP_W, 0.0, 1.0)
                        y_norm = np.clip(cy_warped / WARP_H, 0.0, 1.0)

                        zone = int(x_norm * NUM_ZONES)
                        zone = max(0, min(NUM_ZONES - 1, zone))
                        '''
                        if x_norm != last_x_sent or y_norm != last_y_sent:
                            payload_coords = json.dumps({
                                "type": "player_position",
                                "x": round(float(x_norm), 2),
                                "y": round(float(y_norm), 2),
                                "ts": time.time()
                            })
                            mqtt_client.publish(MQTT_TOPIC_PUB, payload_coords) 
                            last_x_sent, last_y_sent = x_norm, y_norm

                        if zone != last_zone:
                            mqtt_client.publish("servo/control", json.dumps({"zone": zone}))
                            last_zone = zone
                        '''


    if detection_in_frame:
        tracking_ok = True
        lost_frames = 0
    else:
        tracking_ok = False
        lost_frames += 1


    cx_warped_text = f"Warped X: {warped_cx}"
    cy_warped_text = f"Warped Y: {warped_cy}"
    
    warped_width_text = f"Warped width: {WARP_W}"
    warped_heigh_text = f"Warped heigh: {WARP_H}"
    cy_text = f"Y coordinates: {cy}"
    cx_text = f"X coordinates: {cx}"

                #cv2.putText(frame, cx_warped_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    #cv2.putText(frame, cx_normalized_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    #cv2.putText(frame, cy_normalized_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cx_warped_text , (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cy_warped_text , (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cx_text , (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, cy_text , (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Отображение информации
    status_text = f"People: {person_count}  FPS: {int(fps)}"    
    cx_text = f"Normal X: {WARP_W}"

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
                        # Example usage:
            '''
            if x_norm is not None and y_norm is not None:
                
            
                theta, phi, rpm = calculate_launcher_shot_normalized(
                    round(float(x_norm), 2),
                    round(float(y_norm), 2),
                    Y_net=Y_net_computed        # ← вычислено из калибровки
                )
                print(f"Elevation θ: {math.degrees(theta):.2f}°")
                print(f"Horizontal φ: {math.degrees(phi):.2f}°")
                print(f"Wheel RPM: {rpm:.0f}")
                print(f"XNorm{x_norm}")
                print(f"RoundXNorm{round(float(x_norm),2)}")
            '''
            
                

    cv2.imshow("Court Zones", frame)


    inside = (
    warped_cx is not None and
    0 <= warped_cx < WARP_W 
)
    
    
    
    
    # OUTSIDE ALL LOOPS
    if person_count>0:
        lost_frames=0
        tracking_ok=True
    else:
        lost_frames+=1
        



    # ESC для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()