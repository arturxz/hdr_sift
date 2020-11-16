import handleKeypoints
import sys
import cv2
import numpy as np
import math

from numpy import linalg as LA
from utils import gaussian_filter, cart_to_polar_grad, get_grad, quantize_orientation

# consts values
bins = 8
sigma_c = 1.5
radius_c = 3 * sigma_c

# Feature Vector Final List
featVectorList = []

def get_patch_grads(p):
    r1 = np.zeros_like(p)
    r1[-1] = p[-1]
    r1[:-1] = p[1:]

    r2 = np.zeros_like(p)
    r2[0] = p[0]
    r2[1:] = p[:-1]

    dy = r1-r2

    r1[:,-1] = p[:,-1]
    r1[:,:-1] = p[:,1:]

    r2[:,0] = p[:,0]
    r2[:,1:] = p[:,:-1]

    dx = r1-r2

    return dx, dy

def get_histogram_for_subregion(m, theta, num_bin, reference_angle, bin_width, subregion_w):
    hist = np.zeros(num_bin, dtype=np.float32)
    c = subregion_w/2 - .5

    for i, (mag, angle) in enumerate(zip(m, theta)):
        angle = (angle-reference_angle) % 360
        binno = quantize_orientation(angle, num_bin)
        vote = mag

        # binno*bin_width is the start angle of the histogram bin
        # binno*bin_width+bin_width/2 is the center of the histogram bin
        # angle - " is the distance from the angle to the center of the bin 
        hist_interp_weight = 1 - abs(angle - (binno*bin_width + bin_width/2))/(bin_width/2)
        vote *= max(hist_interp_weight, 1e-6)

        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)
        vote *= x_interp_weight * y_interp_weight

        hist[binno] += vote

    return hist

# params: angle in rads, x, y, orige x, orige y, answer axis(0==x,1==y)
def rotateP(o, x, y, ox, oy, ansA):
    ans = 0
    if ansA == 0:
        ans = (x * math.cos(o) - y * math.sin(o)) + ox
    else:
        ans = (x * math.sin(o) + y * math.cos(o)) + oy

    return int(round(ans))

# checking if is out of bounds for gradient calc
def isOut(img, x, y):
    h, w = img.shape[:2]
    return (x <= 0 or x >= w-1 or y <= 0 or y >= h-1)

def formatOut(vec):
	out = ""
	out += str(vec[0:20]) + "\n"
	out += " " + str(vec[20:40]) + "\n"
	out += " " + str(vec[40:60]) + "\n"
	out += " " + str(vec[60:80]) + "\n"
	out += " " + str(vec[80:100]) + "\n"
	out += " " + str(vec[100:120]) + "\n"
	out += " " + str(vec[120:128]) + "\n"
	
	return out
	

##### main #####
#call ex: python3 siftDescriptor.py ../TestImages/100.LDR.surf.kp.txt ../TestImages/100.jpg
kpList = []
# geting keypoints in file
filePath = sys.argv[1]
f = open(filePath, "r")
for line in f:    
    x, y, s, o = line.split()
    kp = handleKeypoints.KeyPoint(int(x), int(y), int(s), int(o))
    kpList.append(kp)
f.close()

imgPath = sys.argv[2]

img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binWidth = int(360/bins)

windowSize = 16
sigma = windowSize/6
radius = int(windowSize/2)

toRadVal = 180.0/np.pi

hist = [[np.zeros(bins, dtype=np.float32) for _ in range(4)] for _ in range(4)]

for kp in kpList:
    i = 0
    cx, cy, s = kp.x, kp.y, kp.scale
    kpDir = kp.dir
    theta = kpDir * np.pi/180.0
    kernel = gaussian_filter(windowSize/6)

    t, l = max(0, cy-windowSize//2), max(0, cx-windowSize//2)
    b, r = min(img.shape[0], cy+windowSize//2+1), min(img.shape[1], cx+windowSize//2+1)
    patch = img[t:b, l:r]

    dx, dy = get_patch_grads(patch)

    if dx.shape[0] < windowSize+1:
        if t == 0: kernel = kernel[kernel.shape[0]-dx.shape[0]:]
        else: kernel = kernel[:dx.shape[0]]
    if dx.shape[1] < windowSize+1:
        if l == 0: kernel = kernel[kernel.shape[1]-dx.shape[1]:]
        else: kernel = kernel[:dx.shape[1]]

    if dy.shape[0] < windowSize+1:
        if t == 0: kernel = kernel[kernel.shape[0]-dy.shape[0]:]
        else: kernel = kernel[:dy.shape[0]]
    if dy.shape[1] < windowSize+1:
        if l == 0: kernel = kernel[kernel.shape[1]-dy.shape[1]:]
        else: kernel = kernel[:dy.shape[1]]

    m, theta = cart_to_polar_grad(dx, dy)
    
    if len(kernel) == 17:
        dx = dx.dot(kernel)
        dy = dy.dot(kernel)
    
    subregion_w = windowSize//4
    featvec = np.zeros(bins * windowSize, dtype=np.float32)
    
    for i in range(0, subregion_w):
        for j in range(0, subregion_w):
            t, l = i*subregion_w, j*subregion_w
            b, r = min(img.shape[0], (i+1)*subregion_w), min(img.shape[1], (j+1)*subregion_w)

            hist = get_histogram_for_subregion(m[t:b, l:r].ravel(), 
                                            theta[t:b, l:r].ravel(), 
                                            bins, 
                                            s, 
                                            binWidth,
                                            subregion_w)
            featvec[i*subregion_w*bins + j*bins : i * subregion_w * bins + (j+1)* bins] = hist.flatten()

    featvec /= max(1e-3, LA.norm(featvec))
    
    # changing interval of vec for [0,1] to [0,255] integer        
    featvec = [int(x*255) for x in featvec]
    featVectorList.append(featvec)

ind = imgPath.rfind('/') + 1
fileName = imgPath[ind:]
print(fileName)
if( fileName.split(".").pop().lower() == "tiff" ):
    fileName = imgPath[ind:-4]
else:
    fileName = imgPath[ind:-3]

with open(fileName+'key', 'w') as f:
    i = 0
    out = str(len(featVectorList)) + " 128\n"
    
    for vec in featVectorList:
        newDir = float(f"{kpList[i].dir}") - 180
        newDir = float(int(math.radians(newDir)*1000)/1000)
        if i == 0:
            out = out + str(kpList[i].y) +" "+ str(kpList[i].x) +" "+ str(kpList[i].scale) +" "+ str(newDir) + "\n"
        else:
            out = str(kpList[i].y) +" "+ str(kpList[i].x) +" "+ str(kpList[i].scale) +" "+ str(newDir) + "\n"
        i+=1
        out = out + ' ' + formatOut(vec) #.replace('\n', ' ')
        
        out = out.replace('[', '').replace(']', '').replace(',', '')
		
        f.write(out)

print ("fim")

## file output format: 
## Each line has a vector descriptor been the first and second position of vector the x and y KP 
## position, respectivly.
