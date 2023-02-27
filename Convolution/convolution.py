import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

weighted_avg_kernel3 = np.array([[1,2,1],
                                [2,4,2],
                                [1,2,1]], dtype=np.float32)

stronger_avg_kernel3 = np.array([[1,3,1],
                                [3,9,3],
                                [1,3,1]])
                            
avg_kernel3 = np.ones((3, 3), np.float32)

weighted_avg_kernel5 = np.array([[1,2,4,2,1],
                               [2,4,8,4,2],
                               [4,8,16,8,4],
                               [2,4,8,4,2],
                               [1,2,4,2,1]], dtype=np.float32)

avg_kernel5 = np.ones((5, 5), np.float32)

laplace_kernel = np.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]])

weighted_avg_kernel7 = np.array([[1,2,4,7,4,2,1],
                                 [2,4,7,14,7,4,2],
                                 [4,7,14,21,14,7,4],
                                 [7,14,21,42,21,14,7],
                                 [4,7,14,21,14,7,4],
                                 [2,4,7,14,7,4,2],
                                 [1,2,4,7,4,2,1]], dtype=np.float32)

sharp_kernel = np.array([[0, -1, 0],
                         [-1, 5,-1],
                         [0, -1, 0]])

sobel_detector_x = np.array([[-1,0,1],
                             [-1,0,1],
                             [-1,0,1]])

sobel_detector_y = np.array([[-1,-1,-1],
                             [ 0, 0, 0],
                             [ 1, 1, 1]])

prewitt_detector_x = np.array([[-1,0,1],
                               [-2,0,2],
                               [-1,0,1]])

prewitt_detector_y = np.array([[-1,-2,-1],
                               [ 0, 0, 0],
                               [ 1, 2, 1]])

def check_image(name):
    img = cv2.imread(name, 0) # read and check image
    if img is None:
        print("error")
    return img

def pad_image(img, struc_size):
    # pad image with (struc size / 2) zeros 
    pad_size = struc_size // 2
    height, width = img.shape[:2]
    padded_img = np.zeros(shape=(height+(2*pad_size), width+(2*pad_size)), dtype=np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            padded_img[i+pad_size, j+pad_size] = img[i,j]
    return padded_img

def rescale_image(self, img):
    height, width = img.shape[:2]
    rescaled_image = np.zeros(shape=(height/2,width/2), dtype=np.uint8)
    for i in range(0, height/2, 2):
        for j in range(0, width/2, 2):
            rescaled_image[i/2,j/2] = img[i,j]
    return rescaled_image

def convolve_image(img, kernel, struc_size):
    height, width = img.shape[:2]
    convoluted_image = np.zeros(shape=(height, width), dtype=np.uint8)
    pad_size = struc_size // 2
    for i in range(1, height-1):
        for j in range(1, width-1):
            for x in range(-pad_size, pad_size+1):
                for y in range(-pad_size, pad_size+1):                    
                    convoluted_image[i,j] += (img[i+x,j+y]*kernel[x,y])
            convoluted_image[i,j] = convoluted_image[i,j] / kernel.size
    return convoluted_image

def straight_edges(img, detector):
    height, width = img.shape[:2]
    straight_image = np.zeros(shape=(height, width), dtype=np.uint8)
    detector_size = int(math.sqrt(detector.size) / 2)
    for i in range(1, height-1):
        for j in range(1, width-1):
            for x in range(-detector_size, detector_size):
                for y in range(-detector_size, detector_size):                    
                    straight_image[i,j] += (img[i+x,j+y]*detector[x,y])
    return straight_image

def combine_edge_strength(horizontal_image, vertical_image):
    height, width = horizontal_image.shape[:2]
    combined_image = np.zeros(shape=(height, width), dtype=np.uint8)
    for i in range(1, height-1):
        for j in range(1, width-1):
            new_value = math.sqrt(horizontal_image[i,j]**2 + vertical_image[i,j]**2)
            combined_image[i,j] = new_value
    return combined_image

def histogram(img, name):
    # Calculate the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.reshape(256)

    # Plot histogram
    plt.bar(np.linspace(0,255,256), hist)
    plt.title(name)
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()

def on_change_thresh(value):
    #T, img_threshold = cv2.threshold(img_combine, value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, img_threshold = cv2.threshold(img_combine, value, 255, cv2.THRESH_BINARY)
    #cv2.namedWindow('Thresholded')
    cv2.imshow('Thresholded', img_threshold)

# main code
img_grey = check_image('kitty.bmp')
cv2.imshow('Original', img_grey)
#cv2.imwrite('grey_image.png', img_grey)

img_pad = pad_image(img_grey, 3)
#cv2.imshow('Padded', img_pad)

img_convolve = convolve_image(img_pad, laplace_kernel, 3)
cv2.imshow('Convolved', img_convolve)
#cv2.imwrite('weighted_convolve_3.png', img_convolve)

#img_w_vertical = straight_edges(img_convolve, prewitt_detector_x)
#cv2.imshow('Vertical', img_w_vertical)

#img_w_horizontal = straight_edges(img_convolve, prewitt_detector_y)
#cv2.imshow('Horizontal', img_w_horizontal)

#img_combine = combine_edge_strength(img_w_horizontal, img_w_vertical)
#cv2.imshow('Combined', img_combine)

#T, img_threshold = cv2.threshold(img_combine, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#ret, img_threshold = cv2.threshold(img_combine, 125, 255, cv2.THRESH_BINARY)
#cv2.namedWindow('Thresholded')
#cv2.imshow('Thresholded', img_threshold)
#cv2.createTrackbar('slider', 'Thresholded', 0, 255, on_change_thresh)

#histogram(img_combine, 'combine')
#histogram(img_convolve, 'convolve')

if cv2.waitKey() == 32:
    cv2.destroyAllWindows()

""" Notes for testing and Report
    - Testing based on 3x3 vs 5x5
    - Testing based on blurring multiple times e.g 1-3
    - Testing weighted VS standard 
    - Testing Sobel vs prewitt
"""