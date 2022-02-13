import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon, )
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # It will print the position of the hand in the frame
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # is for a single hand
        return img

    def findPosition(self, img, handNu=0,draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
           myHand= self.results.multi_hand_landmarks[handNu]

           # Will get the id number and the index
           for id, lm in enumerate(myHand.landmark):
             #print(id,lm)
             # this will get us the height and weight
             h, w, c = img.shape
             #  the position in reference of the center
             cx, cy = int(lm.x * w), int(lm.y * h)
             #print(id, cx, cy)
             lmList.append([id,cx,cy])
             # if you want to landmark bigger a particular point yo can do this
             # if id== 5:
             if draw:
              cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img,draw=False)

        if  len(lmList) != 0:
            print(lmList[4])
            # For showing de fps in the window instead the console

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 100), 3)

        cv2.imshow("Image", img)

        # Is the mod for video in my cam
        cv2.waitKey(1)

if __name__ == "__main__":
    main()