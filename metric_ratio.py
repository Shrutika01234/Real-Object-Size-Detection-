from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import numpy as np
import imutils
import cv2

def midpoint(ptA,ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1])*0.5)

def show_image(title,image,destroy_all=True):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def show_image(title, image, destroy_all=True):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
#image = cv2.resize(image, (1000, 1050))
"""
desired_width = 800
height ,width = image.shape[:2]
aspect_ratio = width/height
new_height = int(desired_width/aspect_ratio)
new_size = (desired_width,new_height)
image = cv2.resize(image,new_size)
"""
show_image('image',image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
show_image('image',gray)

edged = cv2.Canny(gray, 20, 60)
show_image('img',edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

print(len(cnts))


# Draw contours

image_copy = image.copy()
image_copy = cv2.drawContours(image_copy,cnts,-1,(0,255,0),thickness=2)

show_image('image',image_copy)

pixelPerMetric = None 
count = 0
for c in cnts:
    if cv2.contourArea(c) < 100:
        continue
    count += 1

    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    print(box)

    

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelPerMetric is None:
        pixelPerMetric = dB / args["width"]
        
    dimA = dB / pixelPerMetric
    dimB = dA / pixelPerMetric
    print(dimA," ",dimB)
    

    cv2.putText(orig, "{:.1f} cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(orig, "{:.1f} cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    cv2.imshow("Image", orig)
    cv2.waitKey(0)

print("Total contours processed: ", count)
print("pixelPerMetric Ratio ",pixelPerMetric)
print()


