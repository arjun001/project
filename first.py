import cv
import numpy as np

class motion:

    def __init__(self, opt, path='G:\Python\Walking Slowly in Front of People.mp4'):
        '''
           Initializing video handle and output windows(640x480)
        '''
        self.opt = opt
        self.window_size = (640,480)
        if (opt == 'file') or (opt == 1):
            self.video_handle = cv.CaptureFromFile(path)
        else:
            self.video_handle = cv.CaptureFromCAM(0)
            # Following two lines provides control on WebCam capture size (not same as output window size)
            #cv.SetCaptureProperty(self.video_handle,cv.CV_CAP_PROP_FRAME_WIDTH,self.window_size[0])
            #cv.SetCaptureProperty(self.video_handle,cv.CV_CAP_PROP_FRAME_HEIGHT,self.window_size[1])
            
        # Creating Named Windows with resizing capability     
        cv.NamedWindow("Original",cv.CV_WINDOW_NORMAL)
       
        
    def init_track_window(self, frame):
        win_name = 'Tracks'
        cv.NamedWindow(win_name,cv.CV_WINDOW_NORMAL)
        frame_size = cv.GetSize(frame)
        # Sample Grayscale Image
        gray = cv.CreateImage(frame_size, cv.IPL_DEPTH_8U, 1)
        '''Edge Detection for giving background reference to the tracks'''
        canny_thres = 300 if self.opt == 'file' else 150
        # Plain White Image for inverting purposes
        white_image = cv.CloneImage(gray)
        cv.Set(white_image,255)
        # Detecting edge, inverting colors and creating a colored image
        gray_frame = cv.CloneImage(gray)
        cv.CvtColor(frame,gray_frame,cv.CV_RGB2GRAY)
        cv.Smooth(gray_frame, gray_frame, cv.CV_GAUSSIAN, 3, 0)
        cv.Canny(gray_frame,gray_frame,canny_thres,0,3)
        cv.Threshold(gray_frame, gray_frame, 70, 70, cv.CV_THRESH_BINARY)
        cv.AbsDiff(gray_frame, white_image, gray_frame)
        track_image = cv.CreateImage(frame_size, 8, 3)
        cv.CvtColor(gray_frame,track_image,cv.CV_GRAY2RGB)
        
        return track_image, win_name 
           
        
    def detect_motion(self,sensitivity='medium'):
        
        #Finding Video Size from the first frame
        frame = cv.QueryFrame(self.video_handle)
        frame_size = cv.GetSize(frame)
        
        '''Initializing Image Variables(to be used in motion detection) with required types and sizes'''
        # Image containg instantaneous moving rectangles
        color_image = cv.CreateImage(frame_size, 8, 3)
        # Resizing to window size
        color_output = cv.CreateImage(self.window_size, 8, 3)
        # Grey Image used for contour detection
        grey_image = cv.CreateImage(frame_size, cv.IPL_DEPTH_8U, 1)
        # Image storing background (moving pixels are averaged over small time window)
        moving_average = cv.CreateImage(frame_size, cv.IPL_DEPTH_32F, 3)
        # Image for storing tracks resized to window size
        track_output = cv.CreateImage(self.window_size, cv.IPL_DEPTH_8U, 3)    
        track_image, track_win = self.init_track_window(frame)
        
        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a    
    

        first = True
        # Infinite loop for continuous detection of motion
        while True:
            '''########## Pixelwise Detection of Motion in a frame ###########'''
            # Capturing Frame
            color_image = cv.QueryFrame(self.video_handle)
            
            ##### Sensitivity Control 1 #####
            if (sensitivity == 'medium') or (sensitivity == 'low'):
               # Gaussian Smoothing
               cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)

            if first:
                difference = cv.CloneImage(color_image)
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0)
                first = False
            else:
                cv.RunningAvg(color_image, moving_average, .020, None)
                

            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1, 0.0)

            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, difference)
            #cv.ShowImage("BG",difference)

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY)
            
            ##### Sensitivity Control 2 #####
            sens_thres = 90 if (sensitivity == 'low') or (self.opt == 'cam') else 40     
            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, sens_thres, 255, cv.CV_THRESH_BINARY)
            
            '''### Blobing moved adjacent pixels, finding closed contours and bounding rectangles ###'''
            ##### Sensitivity Control 3 #####
            if (sensitivity == 'medium') or (sensitivity == 'low'):
               # Dilate and erode to get people blobs
                ker_size = 20 if self.opt =='file' else 50
                cv.Dilate(grey_image, grey_image, None, ker_size)
                cv.Erode(grey_image, grey_image, None, 3)
            
            storage = cv.CreateMemStorage(0)
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            points = []
            while contour:
                bound_rect = cv.BoundingRect(list(contour))
                polygon_points = cv.ApproxPoly( list(contour), storage, cv.CV_POLY_APPROX_DP )
                
                pt1 = (bound_rect[0], bound_rect[1])
                pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
            
                if (self.opt == 'file'):
                    points.append(pt1)
                    points.append(pt2)
                elif (bound_rect[0]-bound_rect[2] > 20) and (bound_rect[1]-bound_rect[3] > 20):
                    points.append(pt1)
                    points.append(pt2)
                    
                box = cv.MinAreaRect2(polygon_points)
                box2 = cv.BoxPoints(box)
                box3 = np.int0(np.around(box2))
                box4 = totuple(box3)
                box5 = box4 + (box4[0],)
                    
                # Filling the contours in the greyscale image (visual blobs instead of just contours) 
                cv.FillPoly( grey_image, [list(polygon_points), ], cv.CV_RGB(255,255,255), 0, 0 )
                    
                # Following line to draw detected contours as well
                #cv.PolyLine( color_image, [ polygon_points, ], 0, cv.CV_RGB(255,0,0), 1, 0, 0 )
                    
                # Drawing Rectangle around the detected contour
                cv.PolyLine(color_image, [ list(box5)] , 0, (0,255,255),2)
                    
                if  len(points): # (self.opt == 'file') and
                        center1 = (pt1[0]+pt2[0])/2
                        center2 = (pt1[1]+pt2[1])/2
                        cv.Circle(color_image, (center1,center2), 5, cv.CV_RGB(0, 255, 0), -1)
                        rad = 3 if self.opt == 'file' else 5
                        cv.Circle(track_image, (center1,center2), rad, cv.CV_RGB(255, 128, 0), -1)
            
                contour = contour.h_next()
               
            # Uncomment to track centroid of all the moved boxes (only for WebCam)
            '''
            if (self.opt == 'cam') and len(points):
                center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
                cv.Circle(track_image, center_point, 15, cv.CV_RGB(255, 128, 0), -1)
            '''    
            cv.Resize(color_image,color_output,cv.CV_INTER_AREA)
            cv.ShowImage("Original", color_output)
            
            cv.Resize(track_image,track_output,cv.CV_INTER_AREA)
            cv.ShowImage(track_win, track_output)
            
            
            # Listen for ESC key
            c = cv.WaitKey(7) % 0x100
            if (0xFF & c == 27):
               cv.SaveImage('Tracks_img_042_' + sensitivity + '.jpeg',track_output)
               break

if __name__=="__main__":
    t = motion(opt='1')
    t.detect_motion(sensitivity='medium')
    cv.DestroyAllWindows()
    