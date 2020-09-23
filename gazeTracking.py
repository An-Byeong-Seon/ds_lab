import cv2

def draw_eye_location(roi, locations):
    for i in range(len(locations)-1):

        if locations[0] is None or locations[1] is None:
            continue

        cv2.line(roi, tuple(locations[i]), tuple(locations[i+1]), (0, 255, 255), 3)

    return roi

scale = 2
cap = cv2.VideoCapture(0)

list_eye_locaton = []
history_eye_locations = []
isDraw = True

while True:
    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    roi = resized_cropped
    roi = cv2.flip(roi, 1)
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi,(7,7),0)

    _, threshold = cv2.threshold(gray_roi,28,255,cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (255, 0, 221), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        cenX = int((x + (x+w)) / 2)
        cenY = int((y + (y+h)) / 2)
        cv2.circle(roi,(cenX, cenY) , 10, (0, 255, 0), -1)

        if isDraw:
            list_eye_locaton.append((cenX, cenY))
        else:
            history_eye_locations.append(list_eye_locaton.copy())
            list_eye_locaton.clear()

        break

    roi = draw_eye_location(roi, list_eye_locaton)
    for eye_locations in history_eye_locations:
        roi = draw_eye_location(roi, eye_locations)

    cv2.imshow("Bin", threshold)
    cv2.imshow("Gray ROI", gray_roi)
    cv2.imshow("Eye tracking", roi)
    key = cv2.waitKey(1)
    if key == 27: #esc
        break
    elif key == 32: #space bar
        list_eye_locaton.clear()
        history_eye_locations.clear()
    elif key == ord('v'):
        isDraw = not isDraw

cv2.destroyAllWindows()