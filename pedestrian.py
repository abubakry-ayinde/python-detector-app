import cv2

# img_file = 'img.jpg'
video = cv2.VideoCapture('dashca pedestrians.mp4')

#Our pre-trained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


#create a car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)



# Run forever until car stops or something
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding.
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
 
    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)


    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangle around the cars
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


    # Display the image with the faces spotted
    cv2.imshow('Self Driving Car', frame)

    # Dont autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

# Release the VideoCapture object
video.release()


print("code completed")