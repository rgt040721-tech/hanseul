import cv2
from ultralytics import YOLO
import requests
import time
import threading
import os
from collections import deque
import statistics
from picamera2 import Picamera2

FIREBASE_URL = "https://smart-muddangyee-queue-default-rtdb.firebaseio.com/queue/semi_station.json"

# ===== 기본 설정 =====
TARGET_FPS = 3
FRAME_INTERVAL = 1.0 / TARGET_FPS
SEND_INTERVAL = 1.0
STUCK_LIMIT = 30

# ===== 대기 판정 =====
WAIT_THRESHOLD = 2.0    # 이 시간 이상 머물러야 대기 인원
GRACE = 1.5             # 이 시간까지는 가려져도 인원에 포함
FORGET = 6.0            # 이 시간 넘게 안 보이면 기록 삭제
MOVE_WINDOW = 1.5       # 최근 몇 초 동안의 이동을 볼지
MOVE_RATIO = 0.6        # 그 사이 키의 몇 배 넘게 움직이면 '지나가는 사람'

# ===== 숫자 안정화 =====
SMOOTH_WINDOW = 9       # 최근 몇 프레임의 중간값 (3FPS x 3초)
CHANGE_HOLD = 2.0       # 새 숫자가 이 시간 유지돼야 화면 숫자 변경

# ===== 차량 가림 대응 =====
OCCLUSION_RATIO = 0.5   # 보이는 사람이 이 비율 아래로 갑자기 줄면 '가림'
OCCLUSION_HOLD = 4.0    # 가림일 때 이전 숫자를 최대 몇 초 유지

model = YOLO('best.tflite')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
time.sleep(1)

wait_times = {}
total_wait = {}
last_seen = {}
in_roi_state = {}
pos_hist = {}
recent_counts = deque(maxlen=SMOOTH_WINDOW)

displayed = 0
candidate = None
candidate_since = None
occlusion_start = None

shared = {"count": 0}
heartbeat = {"main": time.time(), "send": time.time()}


def sender():
    while True:
        count = shared["count"]
        try:
            r = requests.patch(
                FIREBASE_URL,
                json={"people_count": count, "updated_at": {".sv": "timestamp"}},
                timeout=(3, 3),
            )
            if r.ok:
                print(f"[{time.strftime('%H:%M:%S')}] Firebase OK: {count}", flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Firebase ERROR: {r.status_code}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Firebase FAIL: {e}", flush=True)
        heartbeat["send"] = time.time()
        time.sleep(SEND_INTERVAL)


def watchdog():
    while True:
        time.sleep(5)
        now = time.time()
        if now - heartbeat["main"] > STUCK_LIMIT:
            print("STUCK: main loop -> restart", flush=True)
            os._exit(1)
        if now - heartbeat["send"] > STUCK_LIMIT:
            print("STUCK: sender -> restart", flush=True)
            os._exit(1)


threading.Thread(target=sender, daemon=True).start()
threading.Thread(target=watchdog, daemon=True).start()

print("Camera started", flush=True)

try:
    while True:
        loop_start = time.time()
        frame = picam2.capture_array()

        height, width = frame.shape[:2]
        ROI_BOX = [0, 0, width, height]
        current_time = time.time()

        # iou=0.8 : 많이 겹친 박스도 덜 지움 (겹쳐 선 사람 대응)
        results = model.track(frame, persist=True, tracker="custom_tracker.yaml",
                              classes=[0], iou=0.8, verbose=False)

        visible_now = len(results[0].boxes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                foot_x = int((x1 + x2) / 2)
                foot_y = y2
                head_x = foot_x
                head_y = y1
                box_h = max(y2 - y1, 1)

                in_roi = (ROI_BOX[0] <= foot_x <= ROI_BOX[2]) and (ROI_BOX[1] <= foot_y <= ROI_BOX[3])

                if obj_id not in total_wait:
                    total_wait[obj_id] = 0.0

                last_seen[obj_id] = current_time
                in_roi_state[obj_id] = in_roi

                # 이동량 계산 (머리 위치 기준)
                hist = pos_hist.setdefault(obj_id, deque())
                hist.append((current_time, head_x, head_y))
                while hist and current_time - hist[0][0] > MOVE_WINDOW:
                    hist.popleft()

                moving = False
                if len(hist) >= 2 and current_time - hist[0][0] >= 1.0:
                    dx = head_x - hist[0][1]
                    dy = head_y - hist[0][2]
                    if (dx * dx + dy * dy) ** 0.5 > MOVE_RATIO * box_h:
                        moving = True

                if moving:
                    total_wait[obj_id] = 0.0
                    wait_times[obj_id] = current_time
                elif in_roi:
                    if obj_id in wait_times:
                        total_wait[obj_id] += (current_time - wait_times[obj_id])
                    wait_times[obj_id] = current_time
                else:
                    wait_times[obj_id] = current_time

        # 대기 인원 계산 (GRACE초 이내 가려진 사람 포함)
        valid_waiting_count = 0
        for obj_id in total_wait:
            recently_seen = (current_time - last_seen.get(obj_id, 0)) <= GRACE
            if recently_seen and in_roi_state.get(obj_id) and total_wait[obj_id] >= WAIT_THRESHOLD:
                valid_waiting_count += 1

        # 오래 안 보인 사람 기록 삭제
        keys_to_delete = [k for k in total_wait if (current_time - last_seen.get(k, 0)) > FORGET]
        for k in keys_to_delete:
            total_wait.pop(k, None)
            wait_times.pop(k, None)
            last_seen.pop(k, None)
            in_roi_state.pop(k, None)
            pos_hist.pop(k, None)

        # 차량 가림 감지
        if displayed >= 1 and visible_now < displayed * OCCLUSION_RATIO:
            if occlusion_start is None:
                occlusion_start = current_time
            if current_time - occlusion_start < OCCLUSION_HOLD:
                raw_count = displayed
            else:
                raw_count = valid_waiting_count
        else:
            occlusion_start = None
            raw_count = valid_waiting_count

        # 숫자 안정화 1: 중간값
        recent_counts.append(raw_count)
        smoothed = statistics.median_low(recent_counts)

        # 숫자 안정화 2: 일정 시간 유지돼야 변경
        if smoothed != displayed:
            if candidate != smoothed:
                candidate = smoothed
                candidate_since = current_time
            elif current_time - candidate_since >= CHANGE_HOLD:
                displayed = smoothed
                candidate = None
        else:
            candidate = None

        shared["count"] = displayed
        heartbeat["main"] = time.time()

        elapsed = time.time() - loop_start
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)

except KeyboardInterrupt:
    print("stop")

finally:
    picam2.stop()
