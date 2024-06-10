import cv2

# Our image
img_file = 'img.jpg'

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'



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

print("Code Completed")