import handleKeypoints
import sys
import cv2
import numpy as np
import math
from numpy import linalg as LA
from utils import gaussian_filter, cart_to_polar_grad, get_grad, quantize_orientation

# consts values
bins = 36
sigma_c = 1.5

def fit_parabola(hist, binno, bin_width):
    centerval = binno*bin_width + bin_width/2.

    if binno == len(hist)-1: rightval = 360 + bin_width/2.
    else: rightval = (binno+1)*bin_width + bin_width/2.

    if binno == 0: leftval = -bin_width/2.
    else: leftval = (binno-1)*bin_width + bin_width/2.
    
    A = np.array([
        [centerval**2, centerval, 1],
        [rightval**2, rightval, 1],
        [leftval**2, leftval, 1]])
    b = np.array([
        hist[binno],
        hist[(binno+1)%len(hist)], 
        hist[(binno-1)%len(hist)]])

    x = LA.lstsq(A, b, rcond=None)[0]
    if x[0] == 0: x[0] = 1e-6
    return -x[1]/(2*x[0])


# checking if is out of bounds for gradient calc
def isOut(img, x, y):
    h, w = img.shape[:2]
    return (x <= 0 or x >= w-1 or y <= 0 or y >= h-1)

### function that gets the orientation of a passed KP of Keypoint type
def calcOrientation(img, kp):
    auxList = []
    sigma = sigma_c * kp.scale
    radius = int(2*np.ceil(sigma)+1)
    hist = np.zeros(bins, dtype=np.float32) 

    kernel = gaussian_filter(sigma)

    for i in range(-radius, radius+1):
        y = kp.y + i
        if isOut(img, 1, y):
            continue
        for j in range(-radius, radius+1):
            x = kp.x + j
            if isOut(img, x, 1):
                continue

            mag, theta = get_grad(img, x, y)            
            weight = kernel[i+radius, j+radius] * mag
            
            binn = quantize_orientation(theta, bins) - 1
            hist[binn] += weight
        
    maxBin = np.argmax(hist)
    maxBinVal = np.max(hist)

    kp.setDir(maxBin*10)

    # checking if exist other valeus above 80% of the max
    #print ('->', hist)

    for binno, k in enumerate(hist):
        if binno == maxBin:
            continue
        if k > .85 * maxBinVal:
            nkp = handleKeypoints.KeyPoint(kp.x, kp.y, kp.scale, binno*10)
            auxList.append(nkp)
        
    return auxList


##### main #####
# Ex call:
# python siftKPOrientation.py ../TestImages/dirtest.jpg ../TestImages/ dirtest.manual
imgPath = sys.argv[1]
path = sys.argv[2] 
fileName = sys.argv[3]

#imgPath = '../TestImages/home.jpg'
#path = '/home/welerson/Área de Trabalho/TCC/Implementações/TestImages/'
#fileName = 'home.dog'

# read image and passing to gray scale
img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
imgCopy = img[:]
if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gaussian filter input img
img = cv2.GaussianBlur(img, (5, 5), 1.5)

# getting keypoints in file 
keypoints = handleKeypoints.KeyPointList(path, fileName)

# calculating orientation for keypoints in list of keypoints 
newKpAssigned = []
auxList = []
for kp in keypoints.List:
    auxList = calcOrientation(img, kp)

    px = int(30*(np.cos(np.radians(kp.dir))))
    py = int(30*(np.sin(np.radians(kp.dir))))
    
    #print (kp.dir*10, ':', (kp.x + px, kp.y + py), (kp.x, kp.y))
    cv2.arrowedLine(imgCopy, (kp.x, kp.y), (kp.x + px, kp.y + py), (0, 0, 255), 1)
    for point in auxList:
        px = int(30*(np.cos(np.radians(point.dir))))
        py = int(30*(np.sin(np.radians(point.dir))))
        cv2.arrowedLine(imgCopy, (point.x, point.y), (point.x + px, point.y + py), (0, 0, 255), 1)
    newKpAssigned += auxList

# passando os novos KP para a lista final
for newKp in newKpAssigned:
    keypoints.List.append(newKp)

#cv2.imwrite('kpOrientation.jpg', imgCopy)

with open(fileName+'.kp.txt', 'w') as f:
    for item in keypoints.List:
        f.write(str(item.x)+' '+str(item.y)+' '+str(item.scale)+' '+str(item.dir)+'\n')
