import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
#for printin the lines in the hand
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img =cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #It will print the position of the hand in the frame
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #Will get the id number and the index
            for id, lm in enumerate (handLms.landmark):
               # print(id,lm)
                # this will get us the height and weight
                h, w , c= img.shape
                #the position in reference of the center
                cx, cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                # if you want to landmark bigger a particular point yo can do this
                #if id== 5:
                #   cv2.circle(img,(cx,cy),15,(255,0,50),cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS) # is for a single hand


    #For showing de fps in the window instead the console
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,100),3)

    cv2.imshow("Image", img)

    #Is the mod for video in my cam
    cv2.waitKey(1)
