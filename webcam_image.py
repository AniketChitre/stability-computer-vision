import cv2
from datetime import datetime

cam = cv2.VideoCapture(1) #Starting webcam - number refers to camera source, e.g., 0, 1, 2 

cv2.namedWindow("Logitech C920 - OpenCV App")

cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

focus = 85  # min: 0, max: 255, increment:5
cam.set(28, focus) 

while True: 
    ret, frame = cam.read() 

    if not ret:
        print("Webcam Error - No Frame")
        break

    cv2.imshow("Logitech C920 - OpenCV App", frame)

    k = cv2.waitKey(1)

    if k%256 == 27: #escape key is pressed
        print("Escape hit, closing the app")
        break

    elif k%256 == 32: #space key to take image 
        sample_ID = input("Please enter sample ID: ")
        stability = input("The sample is stable True/False: ") 
        pH_status = input("Pre or post pH adjustment: ")
        img_name = "opencv_%s_%s_%s_%s-pHAdj.png" % (datetime.today().strftime('%d-%m-%Y'),sample_ID, stability, pH_status)
        cv2.imwrite(img_name, frame)
        print("Image captured")

cam.release()  # release camera

cv2.destroyAllWindows() # close pop-up window

