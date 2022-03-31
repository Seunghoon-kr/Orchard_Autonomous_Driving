import cv2
import numpy as np

'''
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) == ord('q'): break

capture.release()
cv2.destroyAllWindows()
'''

cap_image = cv2.imread('image/v1_0.jpg') # my image path
print()
slideSize = 112
(w_width, w_height) = (224,224) # window size
print(cap_image.shape[1])
print(cap_image.shape[0])

CAM = np.zeros(((cap_image.shape[0]//slideSize)*slideSize,(cap_image.shape[1]//slideSize)*slideSize,3))
print(type(CAM))
for x in range(0, (cap_image.shape[0] - 224), slideSize):
    for y in range(0, (cap_image.shape[1] - 224), slideSize):
        window = cap_image[x:(x+w_width), y:(y+w_height), :]
        #print(window.shape)
        
        #print('{} * {} , {} * {}'.format(x,y,x+224,y+224))
        #cv2.imshow("VideoFrame", window)
        
        CAM[x:(x+w_width), y:(y+w_height)] += window


cv2.imwrite('cam.jpg', CAM)
cv2.imshow("CAM",CAM/225)
cv2.waitKey(0)
cv2.destroyAllWindows()

