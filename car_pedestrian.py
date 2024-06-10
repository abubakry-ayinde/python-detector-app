import cv2

# Our Image
img_file = 'img.jpg'
video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Run forever until car stops or something
while True:

    # Read the current frame
    (read_succesful, frame ) = video.read()

    # Safe coding.
    if read_succesful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
 
    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    # Display the image with the faces spotted
    cv2.imshow('Clever Programmer Car Detector', frame)

    # Dont autoclose (Wait here in the code and listen for a key press)
    cv2.waitKey(1)

      
"""

# create opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
# create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)


# Draw rectangle around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with the faces spotted
cv2.imshow('Clever Programmer Car Detector', img)

# Dont autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()
"""
print('code completed')