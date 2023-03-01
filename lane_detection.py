import cv2
import numpy as np
'22 frame per second'
img = cv2.imread("lines.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

edges = cv2.Canny(gray, 75, 150)

cv2.imshow("img",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(edges, 1,np.pi/180,50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.imshow("edges",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''video = cv2.VideoCapture("road_car_view.mp4")
while True:
    ret, frame = video.read()
    yl = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([18,94,140])
    up = np.array([48,255,255])
    mask = cv2.inRange(yl, low, up)
    edges = cv2.Canny(mask, 75, 150)
    
    lines = cv2.HoughLinesP(edges, 1,np.pi/180,50,maxLineGap=50)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1,y1), (x2,y2), (18,94,140), 5)
    
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)
    
    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()'''

