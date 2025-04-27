import cv2

face_cap = cv2.CascadeClassifier("C:/Users/Lenovo/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video_cap.isOpened():
    print("Camera not detected or cannot be opened")
else:
    while True:
        ret, video_data = video_cap.read()
        
        if not ret:
            print("Camera not capturing")
            break  # This break is now inside the loop

        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(
            col,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("video_live", video_data)
        
        # Exit the loop when 'a' is pressed
        if cv2.waitKey(10) == ord("a"):
            break

    # Release the video capture object
    video_cap.release()
    cv2.destroyAllWindows()
