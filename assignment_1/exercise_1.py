from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_two_images(img1, title1, img2, title2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray", vmin=0, vmax=1)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap="gray", vmin=0, vmax=1)
    plt.title(title2)

    plt.show()


def show_three_images(img1, title1, img2, title2, img3, title3):
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap="gray", vmin=0, vmax=1)
    plt.title(title1)

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap="gray", vmin=0, vmax=1)
    plt.title(title2)

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap="gray", vmin=0, vmax=1)
    plt.title(title3)

    plt.show()

# (a) Read the image from the file umbrellas.jpg


I = imread('images/umbrellas.jpg')

# ------------------------------------------------------------------------------------------------

# (b) Convert the loaded image to grayscale.


def convert_to_gray(I):
    height, width, channels = I.shape
    I_gray = np.zeros((height, width))

    for i in range(height):
        for j in range(0, width):
            # sestejemo red, green, blue in delimo z 3 -> grayscale
            I_gray[i][j] = (I[i][j][0] + I[i][j][1] + I[i][j][2]) / 3
    return I_gray


I_gray = convert_to_gray(I)

show_two_images(I, "Original Image", I_gray, "Grayscale Image")

# ------------------------------------------------------------------------------------------------

# (c) Cut and display a specific part of the loaded image.


def get_cutout(height_start, height_end, width_start, width_end, img):
    cutout = np.copy(img[height_start:height_end, width_start:width_end])
    return cutout


def get_gray_cutout(height_start, height_end, width_start, width_end, img):
    cutout = np.copy(img[height_start:height_end, width_start:width_end, 1])
    return cutout


img_cutout = get_cutout(130, 260, 240, 450, I)
img_cutout_gray = get_gray_cutout(130, 260, 240, 450, I)
show_three_images(I, "Original Image", img_cutout,
                  "Image Cutout", img_cutout_gray, "Grayscale Image Cutout")

#  Question: Why would you use different color maps?
# Sometimes we might get darker images / histograms, and it might be better to use a different color map
# to better see some details on the image

# ------------------------------------------------------------------------------------------------

# (d) Write a script that inverts a rectangular part of the image.


def invert_rectangle(img, height_start, height_end, width_start, width_end):
    # invertiramo tako da odstejemo vsak rgb value od 1, ker je scale [0, 1], ce bi bil scale [0, 255] -> potem odstejemo od 255 namesto 1
    inverted_img = np.copy(img)
    inverted_img[height_start:height_end, width_start:width_end] = 1 - \
        inverted_img[height_start:height_end, width_start:width_end]
    return inverted_img


inverted_img = invert_rectangle(I, 130, 260, 240, 450)

show_two_images(I, "Original Image", inverted_img,
                "Inverted part of the image")

# Question: How is inverting a grayscale value defined? -> max_value - pixel_value (1 - pixel_value)

# ------------------------------------------------------------------------------------------------

# (e) Perform a reduction of grayscale levels in the image.


def reduce_grayscale_levels(img, maxValue):
    rescaled_img = np.copy(img)
    rescaled_img *= maxValue
    return rescaled_img


reduced_grayscale_img = reduce_grayscale_levels(I_gray, 0.3)

show_three_images(I, "Original Image", I_gray,
                  "Grayscale Image", reduced_grayscale_img, "Rescaled Grayscale Image")
