import cv2
import os


for _, _, files in os.walk("images"):
    None

n=0
for file in files:
    img = cv2.imread("images/cropp/"+file, cv2.IMREAD_COLOR)

    c_img = img.copy()

    h_img, w_img, channel = img.shape

    h = 0
    while h+224 <= h_img:
        w=0
        while w+224 <= w_img:

            roi = c_img[h:h+224, w:w+224]
            st = "class/%s-%s-%s.jpg"%(file,h,w)
            cv2.imwrite(st, roi)

            w = w +224
        h = h + 224
    n+=1
    print(n)

