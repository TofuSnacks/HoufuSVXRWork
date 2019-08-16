import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


#Python: 3.7.3
#Package Versions:
#opencv-python: 4.1.0.25
#numpy: 1.16.4
#matplotlib: 3.1.0

def custom_convert_2_8bit(img, max_I=0):
    if img.dtype != np.uint8:
        if max_I == 0:
            max_I = np.percentile(img, 99) # Compute the 99th percentile and use that
        conv = 65536/max_I  
        print("Conv is ", conv)
        
        img = ((img*conv)/256).astype(np.uint8)
        return img
    else:
        print('image is already 8 bits, skipping...')
        return img

#-------------------------------------------------Start Code for testing----------------------------------------------------------------------------------------

im = cv2.imread('image0.tif', -1)
#0. 3
#1. 0
#2. 3
#3. 2
#4. 0
#5. 0
#6. 0
imgg = im  #Creates a copy of the original image to overlay and display at the end

original = custom_convert_2_8bit(im)

#-------------------------------------------------End Code for testing----------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------
#process_image_kernal: Takes an image and finds the location of the vias on the image, don't call this function directly
#                      unless you don't need partitioning
#Input:
#img: The image we want to process (usually partitioned for thresholding). Note: This function doesn't partition the images
#x_offset: The x distance from the global origin, 0 by default
#y_offset: The y distance from the global origin, 0 by default
#
#Return:
#List of via locations encoded in ellipses. The x/y positions of these ellipses are in global coordinates
#-----------------------------------------------------------------------------------------------------------------------------
def process_image_kernal(img, pin_radius, x_offset=0, y_offset=0):
    #Calculate the min/max areas we're looking for 
    min_area = (((1.5*pin_radius)**2) * 3.1415926)
    max_area = ((pin_radius*3)**2 )* 3.1415926   
    
    #Blur the image to make it easier to process
    im = cv2.blur(img, (10,10))
    
    #We create our custom threshold for this image
    ratio =  1.3 #Higher = less stuff
    flattened = im.flatten()
    chunk_mean = np.mean(flattened)
    sd = np.std(flattened)
    im[im<chunk_mean-ratio*sd]=0
    im[im>=chunk_mean-ratio*sd]=1
    
    #Finds the contours
    contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    partition_ellipses = []    #Holds the ellipses we want to return
    
    

    
    #Looks for the contours we want
    for c in contours:
        area = cv2.contourArea(c)

        if area>= min_area and area<=max_area and len(c) >=5 :  #This area range is larger than a pin, but smaller enough to take out some noise contours
            #Do the offset conversion here
            i = 0
            for arr in c:
                c[:,0][i][0] = c[:,0][i][0] + x_offset
                c[:,0][i][1] = c[:,0][i][1] + y_offset
                i += 1
            #At this point, the coordinates of the contour should be relative to global 
            
            #We create the ellipse and add it to our return list
            el = cv2.fitEllipse(c)
            pos, sizes, angle = el
            
            #Checks whether the ellipse is correctly shaped by calculating minor/major ratio
            if sizes[0]/sizes[1] > .3:
                partition_ellipses.append(el)
            
            
    #-------------------------------------------------------------
    #im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)      
    #cv2.drawContours(im, contours, -1, (0,255,0), 3)
    #for c in partition_ellipses:
        #cv2.ellipse(im,c,(255,0,0), thickness=9)    
    #plt.figure()
    #plt.imshow(img, cmap='gray')
    
    #plt.figure()
    #plt.imshow(im, cmap='gray')
    
    #plt.show()      
    #-------------------------------------------------------------    
    return partition_ellipses
#-----------------------------------------------------------------------------------------------------------------------------
#process_image: Takes an image and finds the location of the vias on the image
#
#Input:
#im: The image we want to process (usually partitioned for thresholding). Note: This function doesn't partition the images
#partition_size: The size (in pixels) of a partition (we only do square partitions for now). Default:1000
#overlap: The amount of overlap we want in a partition, this is a ratio of partition size. Must be  1< x <2
#brighten: Brightens up the image, on by default
#
#Return:
#List of via locations encoded in ellipses. The x/y positions of these ellipses are in global coordinates 
#-----------------------------------------------------------------------------------------------------------------------------
def process_image(im, pin_diameter, partition_size=1000, overlap=1.5):
    maxheight,maxwidth= im.shape
    
    if overlap < 1 or overlap > 2:
        raise ValueError('Please enter an overlap between 1 and 2')
    
    if im.dtype != np.uint8:
        im = custom_convert_2_8bit(im) #Brightens up the image (very much needed), may change depending on image
        #im *=10
        
    if partition_size <=100:
        print("Your partition size is very small, this could potentially give bad results. Try using larger numbers (or the default value) for better results")
    
    if partition_size > maxheight or partition_size > maxwidth:
        print("Your partition size is larger than at least one of your dimensions. The code will (probalby) run but be prepared for bad results")
     
         
    #Partition up the images, this code divides up our image into 1.5*offset sized pieces, we overlap by that .5*offset each iteration
    w = 0
    h = 0
    vias = [] #Stores all of our via locations, we don't check for duplicates (Future TODO?)
    while h < maxheight:
        w = 0
        while w < maxwidth:
            #print("Looking at x: ", w, " and y: ", h)
            img = im[h:h+int(partition_size*overlap), w:w+int(partition_size*overlap)]
            vias = vias + process_image_kernal(img, pin_diameter/2.0, w, h)
            w = w + partition_size
        h = h + partition_size
    
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    canvas = np.zeros(shape=im.shape)

    canvas.fill(1)    
    
    for c in vias:
        cv2.ellipse(canvas,c,(0,255,0), thickness=-1)
        pos, sizes, angle = c
        #print("x: ", pos[0], "y: ", pos[1])
        
    
    #At this point, we should have our via list completely filled out (probably with some duplicates), so we return it
    return vias, canvas



#pin_x and pin_y are supposed to be local (ROI) coordiantes
def is_covered(ROI, pin_x, pin_y, pin_radius):
    mask = np.zeros(shape=ROI.shape)

    mask.fill(4)

    cv2.circle(mask,(pin_x, pin_y),  pin_radius, (0,255,0), -1)
    
    lo = np.equal(ROI, mask).astype(int)
    lo *= 255

    
    x = np.count_nonzero(lo)
    
    
    cv2.circle(lo,(pin_x, pin_y),  pin_radius, (0,255,0), thickness=5)
    plt.figure()
    plt.imshow(lo, cmap='gray')
    plt.show()
    
    if(x<=0):
        return False

    return True


#-------------------------------------------------Start Code for testing----------------------------------------------------------------------------------------
#We call the image processing function here

import time

t0 = time.time()
#i = 0
#while i<100:
#im = cv2.imread('image4.tif', -1)
vias, img = process_image(im, 40) 
#i +=1
t1 = time.time()
plt.figure()
plt.imshow(img, cmap='gray')
total = t1-t0
print("Time Taken: ", total)

#We overlay our ellipses onto the vias
for c in vias:
    cv2.ellipse(imgg,c,(0,255,0), thickness=9)
    pos, sizes, angle = c
    #print("x: ", pos[0], "y: ", pos[1])
    
imgg = custom_convert_2_8bit(imgg)
##Display image
plt.figure()
plt.imshow(imgg, cmap='gray')

plt.figure()
plt.imshow(original, cmap='gray')


print(is_covered(img,2140,400,40))

#-------------------------------------------------End Code for testing----------------------------------------------------------------------------------------




