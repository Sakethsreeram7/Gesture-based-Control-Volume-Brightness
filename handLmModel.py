
import cv2 as cv
from mediapipe import *
import mediapipe as mp
import numpy as np

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        rlmlist = []

        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) == 2:
                rlmlist.append('both')
            elif len(self.results.multi_hand_landmarks) == 1:
                rlmlist.append(self.results.multi_handedness[0].classification[0].label)

            for n in self.results.multi_hand_landmarks:
                lmList = []
                myHand = n
                for id1, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id1, cx, cy])
                    if draw:
                        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
                rlmlist.append(lmList)

        return rlmlist


