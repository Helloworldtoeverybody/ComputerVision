import cv2

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("fail")
        break

    cv2.imshow("test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()