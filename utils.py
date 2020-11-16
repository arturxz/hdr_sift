import sys
import cv2
import numpy as np
import math
from numpy import linalg as LA

def gaussian_filter(sigma):
	size = 2*np.ceil(3*sigma)+1
	x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
	return g/g.sum()

def cart_to_polar_grad(dx, dy):
    m = np.sqrt(dx*dx + dy*dy)
    theta = ((np.arctan2(dy, dx)+np.pi) * 180/np.pi) - 90
    return m, theta

def get_grad(L, x, y):
    dy = int(L[min(L.shape[0]-1, y+1),x]) - int(L[max(0, y-1),x])
    dx = int(L[y,min(L.shape[1]-1, x+1)]) - int(L[y,max(0, x-1)])
    return cart_to_polar_grad(dx, dy)

def quantize_orientation(theta, num_bins):
    bin_width = 360//num_bins
    return int(np.floor(theta)//bin_width)