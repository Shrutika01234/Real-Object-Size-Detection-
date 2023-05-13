"""
import cv2

# Load the image
img = cv2.imread('object_image.jpg')

# Determine the distance between the camera and the object (in meters)
distance = 1.5

# Determine the dimensions of the object in pixels
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])
object_width_in_pixels = w
object_height_in_pixels = h

# Estimate the focal length (in pixels) based on the sensor size and the lens focal length
sensor_size = (3.68, 2.76) # Sensor size in mm
focal_length = 3.6 # Lens focal length in mm
sensor_resolution = (img.shape[1], img.shape[0]) # Sensor resolution in pixels
focal_length_pixels = (focal_length * sensor_resolution[0]) / sensor_size[0]

# Calculate the real width and height of the object (in meters)
real_width = (object_width_in_pixels * distance) / focal_length_pixels
real_height = (object_height_in_pixels * distance) / focal_length_pixels

# Print the real width and height of the object
print("The real width of the object is {:.2f} meters.".format(real_width))
print("The real height of the object is {:.2f} meters.".format(real_height))

"""

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import numpy as np
import imutils
import cv2

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
distance = 15 # 15 cm 
sensor_size = (16.77, 2.76) # Sensor size in mm
focal_length = 16.98 # Lens focal length in mm
sensor_resolution = (image.shape[1], image.shape[0]) # Sensor resolution in pixels
#focal_length_pixels = (focal_length * sensor_resolution[0]) / sensor_size[0]


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

    #if pixelPerMetric is None:
      #  pixelPerMetric = dB / args["width"]

    #dimA = dB / pixelPerMetric
    #dimB = dA / pixelPerMetric
    x,y,w,h = cv2.boundingRect(c)
    real_width = (w * distance) / (focal_length * sensor_size[0])
    real_height = (h* distance) / (focal_length * sensor_size[0]) 

    cv2.putText(orig, "{:.1f}cm".format(real_width), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(orig, "{:.1f}cm".format(real_height), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    cv2.imshow("Image", orig)
    cv2.waitKey(0)

print("Total contours processed: ", count)
