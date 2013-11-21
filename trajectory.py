import cv2
import numpy as np
import matplotlib.pyplot as plt
''' This script reads the video input and saves the motion array'''

file="MovingBall.mp4"
capture = cv2.VideoCapture(file)
print '\t Width: ',capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print '\t Height: ',capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
print '\t FourCC: ',capture.get(cv2.cv.CV_CAP_PROP_FOURCC)
print '\t Framerate: ',capture.get(cv2.cv.CV_CAP_PROP_FPS)
numframes=capture.get(7)
print '\t Number of Frames: ',numframes

count=0
history = 10
nGauss = 3
bgThresh = 0.6
noise = 15
bgs = cv2.BackgroundSubtractorMOG(history,nGauss,bgThresh,noise)

plt.figure()
plt.hold(True)
plt.axis([0,1280,960,0])
measuredTrack=np.zeros((numframes,2))-1
while count<numframes:
    count+=1
    img2 = capture.read()[1]
    cv2.imshow('Video',img2)
    foremat=bgs.apply(img2)
    cv2.waitKey(100)
    foremat=bgs.apply(img2)
    ret,thresh = cv2.threshold(foremat,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        m= np.mean(contours[0],axis=0)
        measuredTrack[count-1,:]=m[0]
        plt.plot(m[0,0],m[0,1],'ob')
    cv2.imshow('Foreground',foremat)
    cv2.waitKey(80)
capture.release()
print measuredTrack
np.save('motion', measuredTrack)
plt.show()
 ########################################################################
 #######################################################################
''' This script takes the motion array as input, and plots the tracks of motion'''
 
import numpy as np
from pykalman import KalmanFilter
from matplotlib import pyplot as plt

Measured=np.load('motion.npy')
while True:
   if Measured[0,0]==-1.:
       Measured=np.delete(Measured,0,0)
   else:
       break
numMeas=Measured.shape[0]
MarkedMeasure=np.ma.masked_less(Measured,0)

Transition_Matrix=[[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
Observation_Matrix=[[1,0,0,0],[0,1,0,0]]

xinit=MarkedMeasure[0,0]
yinit=MarkedMeasure[0,1]
vxinit=MarkedMeasure[1,0]-MarkedMeasure[0,0]
vyinit=MarkedMeasure[1,1]-MarkedMeasure[0,1]
initstate=[xinit,yinit,vxinit,vyinit]
initcovariance=1.0e-3*np.eye(4)
transistionCov=1.0e-4*np.eye(4)
observationCov=1.0e-1*np.eye(2)
kf=KalmanFilter(transition_matrices=Transition_Matrix,
            observation_matrices =Observation_Matrix,
            initial_state_mean=initstate,
            initial_state_covariance=initcovariance,
            transition_covariance=transistionCov,
            observation_covariance=observationCov)

(filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)
plt.plot(MarkedMeasure[:,0],MarkedMeasure[:,1],'xr',label='measured')
plt.axis([0,1280,960,0])
plt.hold(True)
plt.plot(filtered_state_means[:,0],filtered_state_means[:,1],'ob',label='kalman output')
plt.legend(loc=2)
plt.title("Constant Velocity Kalman Filter")
plt.show()

############################################################
############################################################
''' This scripts gives user the choice of input'''

import cv2

from tkFileDialog   import askopenfilename      

type_input = raw_input('Enter 1 [Saved Video] , 0 [WEBCAM FEED]')

if type_input == '1':
    name = askopenfilename()
    
elif type_input == '0':
    name = cv2.VideoCapture(0)
    winName = "Motion Detection"
else:
    print('Enter Choice Again')
