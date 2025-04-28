import cv2
from ultralytics import YOLO
import time

model = YOLO("best.pt").to("cuda")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model.predict(source=frame, conf=0.6, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]

            print(f"🎯 Phát hiện: {label} (index: {class_id}) - {conf:.2f}")

        annotated_frame = r.plot().copy()

    fps = 1 / (time.time() - start)
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    resized_frame = cv2.resize(annotated_frame, (360, 640))  # (width, height)

    cv2.imshow("Traffic Signs Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()