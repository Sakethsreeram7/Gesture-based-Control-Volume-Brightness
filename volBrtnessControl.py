import cv2 as cv
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import threading
cap=cv.VideoCapture(0)
from handLmModel import handDetector

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

handlmsObj = handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]

minBrightness = 0
maxBrightness = 100


def setVolume(dist,frame):
    cv.circle(frame,(xr1,yr1),15,(255,0,255),cv.FILLED)
    cv.circle(frame, (xr2, yr2), 15, (255, 0, 255), cv.FILLED)
    cv.line(frame,(xr1,yr1),(xr2,yr2),(255,0,255),3)
    vol = np.interp(int(dist), [35, 215], [minVolume, maxVolume])
    volbar=np.interp(dist,[50,250],[400,150])
    volper=np.interp(dist,[50,250],[0,100])
    cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(frame, (50, int(volbar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(frame,f'{int(volper)}%',(40,450),cv.FONT_HERSHEY_COMPLEX,1,(0,250,0),3)
    cv.putText(frame, f'RIGHT-VOLUME', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    volume.SetMasterVolumeLevel(vol, None)


def setBrightness(dist,frame):
    cv.circle(frame, (xr1, yr1), 15, (255, 0, 255), cv.FILLED)
    cv.circle(frame, (xr2, yr2), 15, (255, 0, 255), cv.FILLED)
    cv.line(frame, (xr1, yr1), (xr2, yr2), (255, 0, 255), 3)
    brightness = np.interp(int(dist), [35, 230], [minBrightness, maxBrightness])
    volbar = np.interp(dist, [50, 250], [400, 150])
    briper = np.interp(dist, [50, 250], [0, 100])
    cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(frame, (50, int(volbar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(frame, f'{int(briper)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)
    cv.putText(frame, f'LEFT-BRIGHTNESS', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    sbc.set_brightness(int(brightness))


while True:
    success,frame = cap.read()
    frame = cv.flip(frame, 1)
    frame = handlmsObj.findHands(frame,draw=True)
    lndmrks = handlmsObj.findPosition(frame, draw=False)
    if lndmrks:
        # print(lndmrks[4],lndmrks[8])

        xr1, yr1 = lndmrks[1][4][1], lndmrks[1][4][2]
        xr2, yr2 = lndmrks[1][8][1], lndmrks[1][8][2]

        dist = math.hypot(xr2 - xr1, yr2 - yr1)

        if lndmrks[0] == 'Left':
            setBrightness(dist,frame)
        elif lndmrks[0] == 'Right':
            setVolume(dist,frame)
        elif lndmrks[0] == 'both':
            xl1, yl1 = lndmrks[1][4][1], lndmrks[1][4][2]
            xl2, yl2 = lndmrks[1][8][1], lndmrks[1][8][2]
            distl = math.hypot(xl2 - xl1, yl2 - yl1)

            t1 = threading.Thread(target=setVolume, args=(dist,frame))
            t2 = threading.Thread(target=setBrightness, args=(distl,frame))

            t1.start()
            t2.start()


    cv.imshow("stream", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

