import cv2
import os
import numpy as np
import csv
from datetime import datetime, timedelta
from tkinter import Tk, Label, Button, messagebox
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
import winsound
import time  

RTSP_URL = "rtsp://admin:L2E3F7CA@192.168.1.9:554/cam/realmonitor?channel=1&subtype=0"
DETECTION_SIZE = (320, 320)
PROCESS_EVERY_N_FRAMES = 5  
ENABLE_PREVIEW = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

last_logged = {}
known_embeddings = []
known_names = []
running = False

print("[INFO] Loading face recognition model...")
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
print("[INFO] Model loaded.")

def load_known_faces():
    global known_embeddings, known_names
    print("[INFO] Loading known faces...")
    for file in os.listdir('known_faces'):
        if file.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join('known_faces', file))
            faces = app.get(img)
            if faces:
                known_embeddings.append(faces[0].normed_embedding)
                known_names.append(os.path.splitext(file)[0])
    known_embeddings = np.array(known_embeddings)  
    print(f"[INFO] Total faces loaded: {len(known_names)}")

def save_known_embeddings():
    if known_embeddings is not None and len(known_embeddings) > 0:
        np.save("known_embeddings.npy", known_embeddings)
        np.save("known_names.npy", known_names)
        print("[INFO] Embeddings and names saved to disk.")

def load_or_initialize_embeddings():
    global known_embeddings, known_names
    if os.path.exists("known_embeddings.npy") and os.path.exists("known_names.npy"):
        known_embeddings = np.load("known_embeddings.npy")
        known_names = np.load("known_names.npy")
        print("[INFO] Loaded saved embeddings.")
    else:
        load_known_faces()
        save_known_embeddings()

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def log_recognition(name, detection_time, recognition_time):
    now = datetime.now()
    if name in last_logged and now - last_logged[name] < timedelta(minutes=1):
        return

    last_logged[name] = now
    with open('recognition_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S"), f"{detection_time:.2f} ms", f"{recognition_time:.2f} ms"])

    winsound.Beep(1000, 200)
    print(f"[LOG] {name} @ {now.strftime('%H:%M:%S')} | Detection Time: {detection_time:.2f} ms | Recognition Time: {recognition_time:.2f} ms")

def start_camera():
    global running
    running = True
    vs = VideoStream(RTSP_URL)

    print("[INFO] Camera stream started.")
    frame_count = 0

    while running:
        input_time = time.perf_counter()  
        ret, frame = vs.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            continue

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        detect_start = time.perf_counter()
        faces = app.get(frame)
        detect_end = time.perf_counter()

        recognition_start = time.perf_counter()
        for face in faces:
            emb = face.normed_embedding
            bbox = face.bbox.astype(int)

            if not known_embeddings.size:  
                continue

            sims = cosine_similarity([emb], known_embeddings)[0]
            idx = np.argmax(sims)
            conf = sims[idx]

            name = known_names[idx] if conf > 0.5 else "Unknown"
            recognition_end = time.perf_counter()

            detection_time = (detect_end - detect_start) * 1000  
            recognition_time = (recognition_end - recognition_start) * 1000  

            log_recognition(name, detection_time, recognition_time)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        total_end = time.perf_counter()

        total_time = (total_end - input_time) * 1000  
        print(f"[TIME] Total: {total_time:.2f} ms")

        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if ENABLE_PREVIEW:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vs.stop()
    cv2.destroyAllWindows()

def stop_camera():
    global running
    running = False

def start_gui():
    load_or_initialize_embeddings()

    win = Tk()
    win.title("Face Attendance System")
    win.geometry("400x180")

    Label(win, text="RTSP Face Recognition System", font=("Arial", 16)).pack(pady=20)
    Button(win, text="Start Camera", bg="green", fg="white", width=20, height=2,
           command=lambda: Thread(target=start_camera).start()).pack(pady=5)
    Button(win, text="Stop Camera", bg="red", fg="white", width=20, height=2,
           command=stop_camera).pack(pady=5)

    win.mainloop()

if __name__ == "__main__":
    start_gui() 