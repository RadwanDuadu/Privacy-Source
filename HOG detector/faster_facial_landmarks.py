from imutils.video import VideoStream
import time
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial detector")
detector = dlib.get_frontal_face_detector()

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    #  to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    # calculate the ratio of the width and construct the
    # dimensions
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    #  have a maximum width of 400 pixels, and convert it to
    #  grayscale

    frame = vs.read()
    frame = resize(frame, width=400)
    result_image = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        (bX, bY, bW, bH) = rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
			(255, 255, 0), 5)
        sub_face = frame[bY:bY + bH, bX:bX + bW]
        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
        # merge this blurry rectangle to our final image
        result_image[bY:bY + sub_face.shape[0], bX:bX + sub_face.shape[1]] = sub_face

    cv2.imshow("Frame", result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()