import os
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    root = os.getcwd()
    imgPath = os.path.join(root,'Resistor11.jpg')
    img = cv.imread(imgPath)
    Grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cap = cv.VideoCapture('PosRes.mp4') # video address or 0 for webcam
    resistorCompoent_cascade = cv.CascadeClassifier("cascade15.xml") #haarcascade_frontalface_default.xml/cascadeX.xml
    # Define the codec and create a VideoWriter object
    #fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for video output (XVID is a common choice)
    #out = cv.VideoWriter('cascadeCamRecord.avi', fourcc, 20.0, (640, 480))  # Output file name and settings
    #print("Haar cascade XML file path:", os.path.abspath("cascade15.xml"))

    while True:

        ret, cam = cap.read()
        GrayCam = cv.cvtColor(cam, cv.COLOR_BGR2GRAY)
        Resistor = resistorCompoent_cascade.detectMultiScale(Grayimg, 1.1,5)
        camRecognize = resistorCompoent_cascade.detectMultiScale(GrayCam, 1.1,5)
        
        
        if ret:
            # Display the cam 
            cv.imshow('Webcam', cam)

            # Write the frame to the output file
            #out.write(cam)

            # Press 'q' to exit the loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        for x,y,w,h in camRecognize:
            cv.rectangle(cam,(x,y),((x+w),(y+h)),(100,255,100),2)

        for x,y,w,h in Resistor:
            cv.rectangle(img,(x,y),((x+w),(y+h)),(100,255,100),2)
            

        cv.imshow("img",img)
        cv.imshow("cam",cam)
        
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    #out.release()
    cv.destroyAllWindows()


#start = input('press S to start ')
#start = start.lower()
if __name__ == '__main__':
    main()
