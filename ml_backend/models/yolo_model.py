from ultralytics import YOLO
import cv2
from collections import defaultdict

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)

    def detect(self, frames):
        """
        IEEE paper inspired change:
        - Accepts a sequence of frames instead of a single frame
        - Aggregates detections across time (spatio-temporal reasoning)
        """

        # Allow backward compatibility (single frame input)
        if not isinstance(frames, list):
            frames = [frames]

        aggregated_detections = defaultdict(list)
        annotated_frame = None

        for frame in frames:
            results = self.model(frame, verbose=False)
            result = results[0]

            # Keep last annotated frame for visualization
            annotated_frame = result.plot()

            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                aggregated_detections[class_name].append({
                    "confidence": confidence,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })

        # Aggregate detections across frames
        final_detections = []

        for label, detections in aggregated_detections.items():
            avg_confidence = sum(d["confidence"] for d in detections) / len(detections)

            # Use bbox from the most confident detection
            best_bbox = max(detections, key=lambda d: d["confidence"])["bbox"]

            final_detections.append({
                "label": label,
                "confidence": avg_confidence,
                "bbox": best_bbox
            })

        return annotated_frame, final_detections
