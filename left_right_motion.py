import cv
import numpy

class Object:
    def __init__(self):
        self.capture = cv.CaptureFromCAM(0)
        cv.NamedWindow("Object", 1)

    def run(self):
        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
        run_mean = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)
        diff = None
        movement = []

        while True:
            # Capture frame from webcam
            color_image = cv.QueryFrame(self.capture)

            # Smooth to get rid of false positives
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)

            if not diff:
                # Initialize
                diff = cv.CloneImage(color_image)
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, run_mean, 1.0, 0.0)
            else:
                cv.RunningAvg(color_image, run_mean, 0.020, None)

            # Convert the scale of the moving average.
            cv.ConvertScale(run_mean, temp, 1.0, 0.0)

            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, diff)

            # Convert the image to grayscale.
            cv.CvtColor(diff, grey_image, cv.CV_RGB2GRAY)

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)

            # Dilate and erode to get object blobs
            cv.Dilate(grey_image, grey_image, None, 18)
            cv.Erode(grey_image, grey_image, None, 10)

            # Calculate movements
            store_movement = cv.CreateMemStorage(0)
            contour = cv.FindContours(grey_image, store_movement, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            points = []

            while contour:
                # Draw rectangles
                bound_rect = cv.BoundingRect(list(contour))
                contour = contour.h_next()

                pt1 = (bound_rect[0], bound_rect[1])
                pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                points.append(pt1)
                points.append(pt2)
                cv.Rectangle(color_image, pt1, pt2, cv.CV_RGB(255,0,0), 1)

            num_points = len(points)

            if num_points:
                x = 0
                for point in points:
                    x += point[0]
                x /= num_points

                movement.append(x)

            if len(movement) > 0 and numpy.average(numpy.diff(movement[-30:-1])) > 0:
              print 'Left'
            else:
              print 'Right'

            # Display frame to user
            cv.ShowImage("Object Direction of Motion", color_image)

            # Listen for ESC or ENTER key
            c = cv.WaitKey(7) % 0x100
            if (0xFF & c == 27):
               break

if __name__=="__main__":
    t = Object()
    t.run()