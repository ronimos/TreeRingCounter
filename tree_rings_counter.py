# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:05:10 2019

This script counts tree rings along a drawn line. 
The script opens a file selector and wait for the user to draw 
a line which the script will count the tree ring along.
Click Esc to move from one screen to another. 

@author: Ron Simenhois
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

x1, x2, y1, y2 = 0,0,0,0
timg = None
draw = False

def onmouse(event, x, y, flags, paprm):
    """
    An event call function:
        this function draws  a blue line between the point where the mouse was clicked and the mouse location,
        It saves the end of line points.
    Parameters:
        event: int - the mouse action code
        x, y: int - the curser location on the image
        flags: unused
        params: unused
    returns: None
    """
    global x1,x2,y1,y2, timg, draw
    if draw:
        timg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        x1, x2, y1, y2 = 0,0,0,0
        x1 = x
        y1 = y
        
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(timg, (x1,y1), (x,y), (255,0,0), 2)
        x2,y2=x,y
        draw = False

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if draw==True:
            cv2.line(timg, (x1,y1), (x,y), (255,0,0), 2)
            x2,y2=x,y
                        
def rotate_image(img, p1, p2):
    """
    Calculate rotation angle to transfom a line between two point to be horizontal
    and rotate an image by this angle
    Parameters: 
        img: a numpy array - the image to rotate
        p1, p2: (int,int) - start and end points of the rotation line
    """
    def _rotate_point(p, rotation_matrix):
        """
        A helper function that return point location after linear transformation
        Parameters:
            p:(int, int) - point to transform
            rotation_matrix - transformation matrix
        """
        return rotation_matrix.dot(np.array(p+(1,))).astype(int)

    x1,y1 = p1
    x2,y2 = p2
    if x1==x2:
        rotaton_angle = 90
    else:
        rotaton_angle = np.rad2deg(np.arctan((y1-y2)/(x1-x2)))
    
    #center = (int(abs(x1-x2)), int(abs(y1-y2)))
    rows, cols = img.shape[:2]
    center = (cols/2, rows/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotaton_angle,1)
    
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_matrix[0,0]) 
    abs_sin = abs(rotation_matrix[0,1])

    # calculate the new width and height
    new_w = int(rows * abs_sin + cols * abs_cos)
    new_h = int(rows * abs_cos + cols * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_matrix[0, 2] += new_w/2 - center[0]
    rotation_matrix[1, 2] += new_h/2 - center[1]

    rotated = cv2.warpAffine(img, rotation_matrix,(new_w, new_h))

    rx1, ry1 = _rotate_point(p1, rotation_matrix)
    rx2, ry2 = _rotate_point(p2, rotation_matrix)
    
    return rotated, rx1, ry1, rx2, ry2
    
def count_and_show_results(rotated, rx1, ry1, rx2, ry2):

    line=rotated[ry1-30: ry1+30, min(rx1,rx2):max(rx1,rx2)]
    gray = cv2.cvtColor(line[30:35,:,:], cv2.COLOR_BGR2GRAY)
    intensity = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
    line = cv2.rectangle(line,(0,30),(line.shape[1],35), (0,0,0), 1)
    intensity=intensity.mean(axis=0)
    fig, (axim, axin, axring) = plt.subplots(nrows=3, ncols=1, figsize=(15,8))
    axim.imshow(line[...,::-1], aspect='auto') 
    axim.set_title('Cross section of the tree trunk where the ring are counted')
    axim.axis('off')
    axim.margins(0)
    axin.plot(intensity)
    axin.set_title('Average pixle intencity on the y axis inside the black box')
    threshholds = intensity.mean()-intensity.std() * 1.0
    axin.hlines(threshholds, xmax=-1, xmin=len(intensity), colors='r')
    axin.margins(0)
    ring = (intensity<threshholds).astype(int)
    axring.plot(ring)
    count = (np.diff(ring)>0).sum()
    axring.set_title('Ring markers, total ring count is {}'.format(count))
    axring.margins(0)
    plt.tight_layout()    
    
    
# Select image and read it
Tk().withdraw()
img_file = askopenfilename(filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img = cv2.imread(img_file)

# Set the window and mouse event for line draw with a mouse
cv2.namedWindow('Tree rings', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Tree rings', onmouse)


timg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# Wait for a line to be drawn and update the image with a new line every time the  mouse moves
while True:
    cv2.imshow('Tree rings', timg)
    k = cv2.waitKey(30)
    if k & 0xFF==27:
        cv2.destroyAllWindows()
        break
if (x1,y1)!=(x2,y2):
    timg = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    
    # Rotate the image so the drawn line will be horizontal for easy calculations
    rotated, rx1, ry1, rx2, ry2 = rotate_image(timg, (x1,y1), (x2,y2))
    
    count_and_show_results(rotated, rx1, ry1, rx2, ry2)
