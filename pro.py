#!/usr/bin/env python

import cv2.cv as cv
import cv2 as cv2
import numpy as np



class Target:

    def __init__(self):
        #self.capture = cv.CaptureFromFile("E:\Documents and Settings\Owner\My Documents\Downloads\Walking Slowly in Front of People.mp4")
        self.capture = cv.CaptureFromCAM(0)
        cv.NamedWindow("Target", 1)
        cv.NamedWindow("BG1", 1)
        #cv.NamedWindow("BG2", 1)
        cv.NamedWindow("BG3", 1)
        #while True:
        #    nframes =+ 1

    def run(self):
        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        #nframes =+ 1
        
        frame_size = cv.GetSize(frame)
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        first = True

        while True:
            closest_to_left = cv.GetSize(frame)[0]
            closest_to_right = cv.GetSize(frame)[1]

            color_image = cv.QueryFrame(self.capture)

            # Smooth to get rid of false positives
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)

            if first:
                difference = cv.CloneImage(color_image)
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0)
                first = False
            else:
                cv.RunningAvg(color_image, moving_average, .1, None)
                cv.ShowImage("BG",moving_average)

            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1, 0.0)

            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, difference)
            #cv.ShowImage("BG",difference)

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY)
            cv.ShowImage("BG1", grey_image)

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 40, 255, cv.CV_THRESH_BINARY)
            #cv.ShowImage("BG2", grey_image)

            # Dilate and erode to get people blobs
            cv.Dilate(grey_image, grey_image, None, 8)
            cv.Erode(grey_image, grey_image, None, 3)
            cv.ShowImage("BG3", grey_image)

            storage = cv.CreateMemStorage(0)
            global contour
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

            points = []

            while contour:
                global bound_rect
                bound_rect = cv.BoundingRect(list(contour))
                polygon_points = cv.ApproxPoly( list(contour), storage, cv.CV_POLY_APPROX_DP )
                contour = contour.h_next()

                global pt1, pt2
                pt1 = (bound_rect[0], bound_rect[1])
                pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])

               
                #size control
                if (bound_rect[0]-bound_rect[2] > 10) and (bound_rect[1]-bound_rect[3] > 10):
                

                    points.append(pt1)
                    points.append(pt2)

                    #points += list(polygon_points)
                    global box, box2, box3, box4, box5
                    box = cv.MinAreaRect2(polygon_points)
                    box2 = cv.BoxPoints(box)
                    box3 = np.int0(np.around(box2))
                    box4 = totuple(box3)
                    box5 = box4 + (box4[0],)
                    


                    cv.FillPoly( grey_image, [list(polygon_points), ], cv.CV_RGB(255,255,255), 0, 0 )
                    cv.PolyLine( color_image, [ polygon_points, ], 0, cv.CV_RGB(255,255,255), 1, 0, 0 )
                    cv.PolyLine(color_image, [ list(box5)] , 0, (0,0,255),2)
                    #cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 1)

                    if len(points):
                        #center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
                        center1 = (pt1[0]+pt2[0])/2
                        center2 = (pt1[1]+pt2[1])/2
                        #print center1, center2, center_point
                        #cv.Circle(color_image, center_point, 40, cv.CV_RGB(255, 255, 255), 1)
                        #cv.Circle(color_image, center_point, 30, cv.CV_RGB(255, 100, 0), 1)
                        #cv.Circle(color_image, center_point, 20, cv.CV_RGB(255, 255, 255), 1)
                        cv.Circle(color_image, (center1,center2), 5, cv.CV_RGB(0, 0, 255), -1)

            cv.ShowImage("Target", color_image)


            # Listen for ESC key
            c = cv.WaitKey(7) % 0x100
            if c == 27:
                #cv.DestroyAllWindows()
                break

if __name__=="__main__":
    t = Target()
    t.run()
    cv.DestroyAllWindows()
    