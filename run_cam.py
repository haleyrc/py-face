import cv2
import time
import numpy as np
from datetime import datetime
from math import atan2, degrees

camera = cv2.VideoCapture(0)
camera.set( 3, 1280 )
camera.set( 4, 720 )
avg_frame = None
font = cv2.FONT_HERSHEY_SIMPLEX
frames = {}
current = "img"
text = {
    'img'       : "Unfiltered",
    'blur'      : "Blurred",
    'scaled_avg': "Average Image",
    'color'     : "Motion Detection",
    'diff'      : "Difference Image",
    'gray'      : "Grayscale Difference",
    'blur_gray' : "Grayscale Difference",
    'face_gray' : "Facial Rec Grayscale",
    }

fourcc = cv2.cv.CV_FOURCC( *'XVID' )
out = cv2.VideoWriter( 'output.avi', fourcc, 20.0, (1280,720) )

face_cascade = cv2.CascadeClassifier( '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml' )
# face_cascade = cv2.CascadeClassifier( '/dropbox/Dropbox/hacking/baby_monitor/lbpcascade_profileface.xml' )
print face_cascade
eye_cascade = cv2.CascadeClassifier( '/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml' )

def take_picture():
  ret, frames["img"] = camera.read()
  # cv2.imwrite( "app/static/img/base.png", img )


  frames["blur"] = cv2.blur( frames["img"], (5,5) )
  # cv2.imwrite( "app/static/img/blur.png", blur )

  cv2.accumulateWeighted( frames["img"], frames["avg_frame"], 0.20 )
  frames["scaled_avg"] = cv2.convertScaleAbs( frames["avg_frame"] )
  # cv2.imwrite( "app/static/img/avg.png", scaled_avg )

  frames["diff"] = cv2.absdiff( frames["blur"], frames["scaled_avg"] )
  # cv2.imwrite( "app/static/img/diff.png", diff )

  frames["gray"] = cv2.cvtColor( frames["diff"], cv2.COLOR_RGB2GRAY )
  frames["blur_gray"] = cv2.blur( frames["gray"], (5,5) )
  # cv2.imwrite( "app/static/img/gray.png", blur_gray )

  threshold_bw, bw = cv2.threshold( frames["blur_gray"], 40, 255, cv2.THRESH_BINARY )
  # cv2.imwrite( "app/static/img/thresh.png", bw )

  pixel_count = cv2.countNonZero( bw )
  frames["color"] = cv2.cvtColor( bw, cv2.COLOR_GRAY2RGB )

  if pixel_count > 20:
    cv2.putText( frames["color"], 'Motion Detected', (10,50), font, 0.5, 
                 (0, 0, 255), 1, cv2.CV_AA )

  cv2.putText( frames[current], text[current], (10,700), font, 1, 
               (255,255,255), 2, cv2.CV_AA )
  cv2.putText( frames[current], str(datetime.now()), (970,700), font, 0.6, 
               (255,255,255), 1, cv2.CV_AA )

  frames["face_gray"] = cv2.cvtColor( frames["img"], cv2.COLOR_BGR2GRAY )
  faces = face_cascade.detectMultiScale( frames["face_gray"], scaleFactor=1.3, 
                                         minNeighbors=2, minSize=(200,200), 
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE )

  if len(faces) != 0:
    for (x,y,w,h) in faces:
      cv2.rectangle( frames[current], (x,y), (x+w,y+h), (255,0,0), 2 )
      roi_gray = frames["face_gray"][y:y+h, x:x+w]
      roi_color = frames[current][y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale( roi_gray, scaleFactor=1.3, minNeighbors=2, 
                                           minSize=(50,50), 
                                           flags=cv2.cv.CV_HAAR_SCALE_IMAGE )
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle( roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2 )
    '''
  else:
    eyes = eye_cascade.detectMultiScale( frames["face_gray"], scaleFactor=1.3, 
                                         minNeighbors=2, minSize=(50,50), 
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE )

    if len(eyes) == 2:
      (ex_1, ey_1, ew_1, eh_1) = eyes[0]
      (ex_2, ey_2, ew_2, eh_2) = eyes[1]
      eye1_mid = ( (ex_1 + (ew_1 / 2)), (ey_1 + (eh_1 / 2)) )
      eye2_mid = ( (ex_2 + (ew_2 / 2)), (ey_2 + (eh_2 / 2)) )
      cv2.line( frames[current], eye1_mid, eye2_mid, (0,0,255), 2 )
      angle = degrees( atan2( eye1_mid[1] - eye2_mid[1], eye1_mid[0] - eye2_mid[0] ) )
      angle = angle + 90.
      the_text = "Head turned " + str(abs(angle)) + " to the " + ( "right" if angle > 0 else "left" )
      cv2.putText( frames[current], the_text + "deg", (10,300), font, 1, (0,0,255),
                   2, cv2.CV_AA )
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle( frames[current], (ex,ey), (ex+ew,ey+eh), (0,255,0), 2 )
    '''


  # cv2.imwrite("app/static/img/color.png", color )
  out.write( frames[current] )

  cv2.imshow( 'frame', frames[current] )

ret, frames["img"] = camera.read()
frames["avg_frame"] = np.float32( frames["img"] )

while(camera.isOpened()):
  take_picture()

  key = 0xFF & cv2.waitKey(1)

  if key == ord('q'):
    break
  elif key == ord('b'):
    current = "blur"
  elif key == ord('a'):
    current = "scaled_avg"
  elif key == ord('c'):
    current = "color"
  elif key == ord('d'):
    current = "diff"
  elif key == ord('g'):
    current = "blur_gray"
  elif key == ord('i'):
    current = "img"
  elif key == ord('f'):
    current = "face_gray"

camera.release()
out.release()
cv2.destroyAllWindows()

'''
while True:
  time.sleep( 100.0 / 1000.0 )
  take_picture()
'''
