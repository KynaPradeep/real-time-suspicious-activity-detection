from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")  

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("YOLO Detection Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        results = model(frame, verbose=False)

        result = results[0]

        annotated_frame = result.plot() 

        person_detected = False
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name == "person":
                person_detected = True

        if person_detected:
            cv2.putText(
                annotated_frame,
                "Person detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
