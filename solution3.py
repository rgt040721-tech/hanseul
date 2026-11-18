import cv2
from ultralytics import YOLO
import requests
import time
import threading
import os
import json
import numpy as np
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

# ===== 천막 / 혼잡도 =====
FRONT_ROW = 5           # 천막 앞줄 칸 수
TENT_CAPACITY = 20      # 천막 최대 인원 (5줄 x 4명)
LEVEL_BUSY = 10         # 천막 뒤 이 인원 이상이면 '혼잡'
LEVEL_VERY_BUSY = 20    # 천막 뒤 이 인원 이상이면 '매우 혼잡'
# level: 0=원활, 1=보통, 2=혼잡, 3=매우 혼잡

# ===== 숫자 안정화 =====
SMOOTH_WINDOW = 9       # 최근 몇 프레임의 중간값 (3FPS x 3초)
CHANGE_HOLD = 2.0       # 새 상태가 이 시간 유지돼야 변경

# ===== 차량 가림 대응 =====
OCCLUSION_RATIO = 0.5   # 보이는 사람이 이 비율 아래로 갑자기 줄면 '가림'
OCCLUSION_HOLD = 4.0    # 가림일 때 이전 상태를 최대 몇 초 유지

# ===== 천막 구역 (zone.json 에서 읽음) =====
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zone.json')) as f:
    TENT = np.array(json.load(f)['tent'], dtype=np.float32)


def in_tent(x, y):
    return cv2.pointPolygonTest(TENT, (float(x), float(y)), False) >= 0


def evaluate(front, back):
    """보이는 앞줄 / 천막 뒤 인원으로 혼잡도와 추정 인원 계산"""
    if back <= 0:
        level = 0
        est = min(front, FRONT_ROW)
        ranged = front >= FRONT_ROW   # 앞줄 꽉 참 → 뒷줄은 안 보이므로 '5~20명'
    else:
        if back >= LEVEL_VERY_BUSY:
            level = 3
        elif back >= LEVEL_BUSY:
            level = 2
        else:
            level = 1
        est = TENT_CAPACITY + back
        ranged = False
    return level, est, ranged


model = YOLO('best.tflite')

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
time.sleep(1)

wait_times = {}
total_wait = {}
last_seen = {}
in_tent_state = {}
pos_hist = {}
recent_front = deque(maxlen=SMOOTH_WINDOW)
recent_back = deque(maxlen=SMOOTH_WINDOW)

displayed = {"front": 0, "back": 0}
candidate = None
candidate_since = None
occlusion_start = None

shared = {"state": (0, 0, False, 0, 0)}   # level, est, ranged, front, back
heartbeat = {"main": time.time(), "send": time.time()}


def sender():
    while True:
        level, est, ranged, front, back = shared["state"]
        try:
            r = requests.patch(
                FIREBASE_URL,
                json={
                    "people_count": est,
                    "level": level,
                    "ranged": ranged,
                    "front_count": front,
                    "back_count": back,
                    "updated_at": {".sv": "timestamp"},
                },
                timeout=(3, 3),
            )
            if r.ok:
                print(f"[{time.strftime('%H:%M:%S')}] Firebase OK: level={level} est={est} front={front} back={back}", flush=True)
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

                if obj_id not in total_wait:
                    total_wait[obj_id] = 0.0

                last_seen[obj_id] = current_time
                in_tent_state[obj_id] = in_tent(foot_x, foot_y)

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
                else:
                    if obj_id in wait_times:
                        total_wait[obj_id] += (current_time - wait_times[obj_id])
                wait_times[obj_id] = current_time

        # 구역별 대기 인원 (GRACE초 이내 가려진 사람 포함)
        raw_front = 0
        raw_back = 0
        for obj_id in total_wait:
            recently_seen = (current_time - last_seen.get(obj_id, 0)) <= GRACE
            if recently_seen and total_wait[obj_id] >= WAIT_THRESHOLD:
                if in_tent_state.get(obj_id):
                    raw_front += 1
                else:
                    raw_back += 1

        # 오래 안 보인 사람 기록 삭제
        keys_to_delete = [k for k in total_wait if (current_time - last_seen.get(k, 0)) > FORGET]
        for k in keys_to_delete:
            total_wait.pop(k, None)
            wait_times.pop(k, None)
            last_seen.pop(k, None)
            in_tent_state.pop(k, None)
            pos_hist.pop(k, None)

        # 차량 가림 감지: 보이는 사람이 갑자기 확 줄면 이전 상태 유지
        shown_total = displayed["front"] + displayed["back"]
        if shown_total >= 1 and visible_now < shown_total * OCCLUSION_RATIO:
            if occlusion_start is None:
                occlusion_start = current_time
            if current_time - occlusion_start < OCCLUSION_HOLD:
                raw_front = displayed["front"]
                raw_back = displayed["back"]
        else:
            occlusion_start = None

        # 숫자 안정화 1: 구역별 중간값
        recent_front.append(raw_front)
        recent_back.append(raw_back)
        sm_front = statistics.median_low(recent_front)
        sm_back = statistics.median_low(recent_back)

        # 숫자 안정화 2: 일정 시간 유지돼야 변경
        new = (sm_front, sm_back)
        cur = (displayed["front"], displayed["back"])
        if new != cur:
            if candidate != new:
                candidate = new
                candidate_since = current_time
            elif current_time - candidate_since >= CHANGE_HOLD:
                displayed["front"], displayed["back"] = new
                candidate = None
        else:
            candidate = None

        level, est, ranged = evaluate(displayed["front"], displayed["back"])
        shared["state"] = (level, est, ranged, displayed["front"], displayed["back"])
        heartbeat["main"] = time.time()

        elapsed = time.time() - loop_start
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)

except KeyboardInterrupt:
    print("stop")

finally:
    picam2.stop()
