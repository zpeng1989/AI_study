

import cv2

img = cv2.imread('lena.jpg',0)

cv2.imwrite('lena_gray.jpg',img)
