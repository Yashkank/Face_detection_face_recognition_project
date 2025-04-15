import cv2
import numpy as np
import os
import csv
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

last_logged = {}

def log_recognition(name):
    current_time = datetime.now()
    if name in last_logged:
        time_diff = current_time - last_logged[name]
        if time_diff < timedelta(minutes=1):  
            return
    last_logged[name] = current_time
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open('recognition_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])
    print(f"[LOGGED] {name} at {timestamp}")

print("[INFO] Loading model...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) 
app.prepare(ctx_id=0)
print("[INFO] Model loaded.")

known_embeddings = []
known_names = []

print("[INFO] Loading known faces...")
for file in os.listdir('known_faces'):
    if file.endswith(('.jpg', '.png')):
        img_path = os.path.join('known_faces', file)
        img = cv2.imread(img_path)
        faces = app.get(img)

        if faces:
            emb = faces[0].embedding
            name = os.path.splitext(file)[0]
            known_embeddings.append(emb)
            known_names.append(name)
        else:
            print(f"[WARN] No face detected in {file}")

print(f"[INFO] Loaded {len(known_names)} known faces.")

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.embedding

        similarities = cosine_similarity([emb], known_embeddings)[0]
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]

        if confidence > 0.5:
            name = known_names[best_match_index]
            log_recognition(name)
        else:
            name = "Unknown"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Recognition - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()