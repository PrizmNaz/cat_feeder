import os
import cv2

def main():
    device = os.environ.get("VIDEO_DEVICE", "/dev/video2")
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        raise ("Can't open the camera {device}")
    else:
        print ("Cap {device} is opened")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Capture failed")
        else:
            cv2.imshow("Cam.test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()