import cv2
import datetime

def differential_Image(d0, d1, d2):

  diff1 = cv2.absdiff(d2, d1)

  diff2 = cv2.absdiff(d1, d0)

  return cv2.bitwise_and(diff1, diff2)

 

cap = cv2.VideoCapture(0)
winName = "Motion Detection"

cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)

differ = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

d = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

diff_1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

 

while True:

  cv2.imshow( winName, differential_Image(differ, d, diff_1) )
  x = cv2.threshold(differential_Image(differ, d, diff_1), 0, 128, cv2.THRESH_OTSU)
 

  # Read next image

  diff = d

  d = diff_1

  diff_1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

  if cv2.countNonZero(diff_1) > x:
    cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', differential_Image(diff, d, diff_1))

  key = cv2.waitKey(10)

  if key == 20:

    cv2.destroyWindow(winName)

    break
print "Check Motion Images in Directory"