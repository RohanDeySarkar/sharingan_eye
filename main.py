import cv2

faceCascadeFilePath = "haarcascade_frontalface_default.xml"
eyeCascade = "haarcascade_eye.xml"


faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
eyeCascade = cv2.CascadeClassifier(eyeCascade)

imgS = cv2.imread('sharingan.png', -1)

orig_mask = imgS[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)

imgS = imgS[:,:,0:3]

origHeight, origWidth = imgS.shape[:2]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        
        )
    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eye = eyeCascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eye:
                        
            sharingan = cv2.resize(imgS, (ew, eh), interpolation = cv2.INTER_AREA)

            mask = cv2.resize(orig_mask, (ew, eh), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (ew, eh), interpolation = cv2.INTER_AREA)
            
            roi = roi_color[ey:ey + eh, ex:ex + ew]
            
            roi_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
            roi_fg = cv2.bitwise_and(sharingan, sharingan, mask = mask)
            
            dst = cv2.add(roi_bg, roi_fg)

            roi_color[ey:ey + eh, ex:ex + ew] = dst

            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
        
































        
    

