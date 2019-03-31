from copy import deepcopy

import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt

def find_template_size(template,trainimage):
    """
    calculates the ratio of bounding box to template image size.
    :param template:  template image numpy array
    :param trainimage: screenshot image numpy array
    :return: returns the ratio of size of detected window to that of the template.
    """
    template = cv2.Canny(template[:, :, 0], 50, 200) | cv2.Canny(template[:, :, 1], 50, 200) | cv2.Canny(
        template[:, :, 2], 50, 200)
    (tH, tW) = template.shape[:2]
    # cv2.imshow("Template", template1)
    # cv2.imshow("template new", template)
    gray = trainimage
    image = trainimage
    found = None
    for scale in np.linspace(0.1, 2.0, 80)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized[:, :, 0], 50, 200) | cv2.Canny(resized[:, :, 1], 50, 200) | cv2.Canny(
            resized[:, :, 2], 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            # print(r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # print(startX - endX, startY - endY)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # cv2.imshow("Image", image)
    cv2.imwrite("bounding_box_size.png" , image )
    cv2.waitKey(1)
    return r

def generate_r(x,y,x1,y1,a,b,a1,b1):
    return  np.sqrt(((x1-x)*(x1-x) + (y1-y)*(y1-y))/((a1-a)*(a1-a) + (b1-b)*(b1-b)))

def histogram(image):
    histr1 = cv2.calcHist([image], [0], None, [256], [0, 256])
    histr2 = cv2.calcHist([image], [1], None, [256], [0, 256])
    histr3 = cv2.calcHist([image], [2], None, [256], [0, 256])
    histr1 = (histr1 / np.mean(histr1))
    histr2 = (histr2 / np.mean(histr2))
    histr3 = (histr3 / np.mean(histr3))

    return (histr1,histr2,histr3)


OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA),
    ("Chi-Squared", cv2.HISTCMP_CHISQR))

# loop over the comparison methods

templatefn = "joystick.png"
queryfn = "game3.png"

MIN_MATCH_COUNT = 10
img1 = cv2.imread(queryfn,0)          # queryImage
img2 = cv2.imread(templatefn,0) # trainImage

template_alpha = cv2.imread(templatefn,cv2.IMREAD_UNCHANGED)
queryimg = cv2.imread(queryfn)
template = cv2.imread(templatefn)
r = find_template_size(template,queryimg)
wrong = cv2.imread("bubble_8.png")

img2 = imutils.resize(img2, width=int((img2.shape[1]) * r))

histr1,histr2,histr3 = histogram(template)
plt.plot(histr1,'r')
plt.plot(histr2,'g')
plt.plot(histr3,'b')
plt.savefig("templatergb.png")
plt.show()
# originalpts = []
# for x,y,z in zip(histr1,histr2,histr3):
#     originalpts.append((x,y,z).index(max(x,y,z)))
hist = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
h1 = cv2.normalize(hist, hist).flatten()

hist = cv2.calcHist([wrong], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
h2 = cv2.normalize(hist, hist).flatten()


for (methodName, method) in OPENCV_METHODS:
    results = {}
    reverse = False
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    d = cv2.compareHist(h1, h1, method)
    print(f"starting {method}  result is {d}")


for (methodName, method) in OPENCV_METHODS:
    results = {}
    reverse = False
    if methodName in ("Correlation", "Intersection"):
        reverse = True
    d = cv2.compareHist(h1, h2, method)
    print(f"wrong  {method}  result is {d}")

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance and m.distance < 100 :
        good.append([m])
        matchesMask[i]=[1,0]
length = len(good)
draw_params = dict(matchColor = (0,255,0),
                   flags = 2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite("sift_matches.png",img3)

print("no of good points ",len(good))


for x in range(length):
    for y in range(x+1,length):
        if (good[x][0].trainIdx == good[y][0].trainIdx or kp2[good[x][0].trainIdx].pt[0] == kp2[good[y][0].trainIdx].pt[0]
                or kp2[good[x][0].trainIdx].pt[1] == kp2[good[y][0].trainIdx].pt[1]):
            continue
        r = generate_r(kp1[good[x][0].queryIdx].pt[0], kp1[good[x][0].queryIdx].pt[1], kp1[good[y][0].queryIdx].pt[0]
                       , kp1[good[y][0].queryIdx].pt[1], kp2[good[x][0].trainIdx].pt[0], kp2[good[x][0].trainIdx].pt[1],
                       kp2[good[y][0].trainIdx].pt[0], kp2[good[y][0].trainIdx].pt[1])
        print("r is ", r)

        resized_template_alpha_img = imutils.resize(template_alpha, width=int((template_alpha[:, :, 3].shape[1]) * r))
        resized_template_alpha_img = resized_template_alpha_img[:, :, 3]
        x1, y1 = resized_template_alpha_img.shape
        StartX, StartY = int(kp1[good[x][0].queryIdx].pt[0] - r * (kp2[good[x][0].trainIdx].pt[0])), int(
            kp1[good[x][0].queryIdx].pt[1] - r * (kp2[good[x][0].trainIdx].pt[1]))
        if (StartX < 0):
            StartX = 0
        if (StartY < 0):
            StartY = 0

        print(StartX, StartY, StartX + x1, StartY + y1)
        print(StartX, StartY, StartX + x1, StartY + y1)


        cropped_img = deepcopy(queryimg[StartY:StartY + y1, StartX: StartX + x1])

        histr1, histr2, histr3 = histogram(cropped_img)
        plt.plot(histr1, 'r')
        plt.plot(histr2, 'g')
        plt.plot(histr3, 'b')
        plt.show()
        hist = cv2.calcHist([cropped_img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        h2 = cv2.normalize(hist, hist).flatten()
        d = 0
        for (methodName, method) in OPENCV_METHODS:
            results = {}
            reverse = False
            if methodName in ("Correlation", "Intersection"):
                reverse = True
            d = cv2.compareHist(h1, h2, method)
            print(f" {method}  result is {d}")
        cv2.imshow("crop00", cropped_img)
        cv2.waitKey(1500)
        if (d > 1.5):
            continue
        cv2.imwrite("cropped_image_before.png", cropped_img)
        cv2.imwrite("templateimg.png",template)
        # cv2.imshow("crop00", cropped_img)
        # cv2.waitKey(1500)

exit(0)



if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

