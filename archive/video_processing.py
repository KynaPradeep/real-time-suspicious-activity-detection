import cv2
from collections import deque
from models.yolo_model import YOLODetector

# IEEE paper inspired change:
# Use fixed-length frame sequences instead of single-frame processing
FRAME_SEQUENCE_LENGTH = 16   # can say "inspired by paper's 20-frame sequence"

# IEEE paper inspired change:
# Store recent predictions for temporal smoothing
PREDICTION_WINDOW = 5

def run_video_detection():
    detector = YOLODetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Real-time Detection Running. Press 'q' to quit.")

    frame_buffer = []
    prediction_history = deque(maxlen=PREDICTION_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read error")
            break

        frame_buffer.append(frame)

        # Run detection only when sequence buffer is full
        if len(frame_buffer) == FRAME_SEQUENCE_LENGTH:
            annotated_frame, detections = detector.detect(frame_buffer)

            # Count people across the frame sequence
            people_count = sum(1 for d in detections if d["label"] == "person")

            # Store prediction for temporal smoothing
            is_suspicious = people_count >= 2
            prediction_history.append(is_suspicious)

            # Temporal aggregation (majority voting)
            suspicious_votes = sum(prediction_history)
            final_decision = suspicious_votes >= (len(prediction_history) // 2 + 1)

            if final_decision:
                cv2.putText(
                    annotated_frame,
                    f"Suspicious Activity Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

            cv2.imshow("Real-Time Detection", annotated_frame)

            # Clear buffer after processing sequence
            frame_buffer.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_detection()
