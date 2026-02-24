import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Motion Detection Started. Press 'q' to quit.")

    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (640, 480))
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    while True:
        ret, frame2 = cap.read()
        frame2 = cv2.resize(frame2, (640, 480))

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        diff = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < 800:  
                continue

            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if motion_detected:
            cv2.putText(frame2, "MOTION DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Motion Detection", frame2)

        gray1 = gray2  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
