import cv2 as cv
import sys
import numpy as np
import math



callBackImg = None
objectLocations = []
objectAngles = []

#GREEN
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#LIGHT BLUE LEGOBLUE
blueLower = (100, 90, 100)
blueUpper = (120, 220, 250)
#PING PONG BALL ORANGE
orangeLower = (10, 160, 200)
orangeUpper = (30, 210, 255)


def areaOfTire():
    orig = cv.VideoCapture(2)
    # Convert to graycsale

    img_gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)

    # Blur the image for better edge detection

    # img_blur = cv.GaussianBlur(img_gray, (105,105),cv.BORDER_DEFAULT)

    (height, width) = img_gray.shape
    scale = 1
    resize = cv.resize(img_gray, (int(width / scale), int(height / scale)))
    img_blur = cv.GaussianBlur(resize, (5, 5), cv.BORDER_DEFAULT)
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)


    EDGES = edges

    cv.imshow("Display window", edges)


    while True:
        k = cv.waitKey(1)
        if k == ord('s'):
            cv.imwrite('testSave.png', callBackImg)
        elif k == ord('q'):
            break
    return edges

#ConfigureOverheadCamera is called when the system is being set up.
#It returns the centroid of the playing field so that the camera knows where the
#base of the arm is in relation to the pixels in the picture.
def configureOverheadCamera():
    global callBackImg
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame exiting ...")
            break
        # Our operations on the frame come here
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        # construct mask for given colour
        greenmask = cv.inRange(hsv, greenLower, greenUpper)
        greenmask = cv.erode(greenmask, None, iterations=2)
        greenmask = cv.dilate(greenmask, None, iterations=2)
        
        bluemask = cv.inRange(hsv, blueLower, blueUpper)
        bluemask = cv.erode(bluemask, None, iterations=2)
        bluemask = cv.dilate(bluemask, None, iterations=2)
        
        orangemask = cv.inRange(hsv, orangeLower, orangeUpper)
        orangemask = cv.erode(orangemask, None, iterations=2)
        orangemask = cv.dilate(orangemask, None, iterations=2)

	# find contours in the masks
        greencnts = cv.findContours(greenmask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
        bluecnts = cv.findContours(bluemask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
        orangecnts = cv.findContours(orangemask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
        #greencnts = cv.drawContours(frame,greencnts)
        #bluecnts = cv.drawContours(frame,bluecnts)
        #orangecnts = cv.drawContours(frame,orangecnts)
        
        # Display the resulting frame
        cv.imshow('frame', greenmask)
        if cv.waitKey(1) == ord('q'):
            break
        
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_threshold = cv.inRange(frame_HSV, (0,0,0), (40,40,40))

    # Image Opening
    kernel = np.ones((1, 1), np.uint8)
    opening = cv.morphologyEx(frame_threshold, cv.MORPH_OPEN, kernel)

    # Make a playing field that is 1 where the field is and 0 everywhere else
    playingField = np.array(opening)
    playingField[playingField == 255] = 1


    # Find the moments of the image (the intensity of each pixel)
    moments = cv.moments(opening)
    # Find the centroid of the intensity pixels
    if moments["m00"] != 0:
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
    else:
        x, y = 0, 0


    # Place a circle as the centroid
    cv.circle(frame, (x, y), 10, (255, 255, 0), -1)
    cX = x
    cY = y
    cv.imshow("Display window", frame)


    while True:
        k = cv.waitKey(1)
        if k == ord('s'):
            cv.imwrite('testSave.png', callBackImg)
        elif k == ord('q'):
            break

    #centroid of the playing field
    return (x,y),playingField

def findObjects(edges):
    orig = cv.VideoCapture(2)

    (height, width, channels) = orig.shape
    scale = 1
    resize = cv.resize(orig, (int(width / scale), int(height / scale)))

    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    frame_threshold = cv.inRange(gray, (60), (180))

    output = cv.connectedComponentsWithStats(frame_threshold, 8, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    area = 0
    x, y = 0, 0

    for i in range(0, numLabels):
        isIn = False
        x, y = centroids[i]
        x = int(x)
        y = int(y)
        if stats[i, cv.CC_STAT_AREA] > 3000 and stats[i, cv.CC_STAT_AREA] < 15000:
            for i in range(-25,25):
                for n in range(-25,25):
                    if edges[y+i,x+n] == 255:
                        isIn = True
            if isIn:
                cv.circle(resize, (int(x), int(y)), 10, (255, 255, 0), -1)
                objectLocations.append((x,y))
    cv.imshow("Display window", resize)
    while True:
        k = cv.waitKey(1)
        if k == ord('s'):
            cv.imwrite('testSave.png', gray)
        elif k == ord('q'):
            break

#Finds all the angles to the objects on the playing field
def findAngles():
    for point in objectLocations:
        x,y = point
        theta = math.degrees(math.atan2(y - cY, x - cX))
        theta = abs(theta)
        objectAngles.append(theta)


if __name__ == '__main__':
    (x,y),playingField = configureOverheadCamera()
    print("Centroid: ",x,y)
    cX = x
    cY = y
    edges = areaOfTire()
    findObjects(edges)
    findAngles()
    print("Object Locations: ", objectLocations)
    print("Object Angles: ", objectAngles)