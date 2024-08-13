from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


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


def create_binary_mask(img, threshold):
    binary_img = np.copy(img)
    binary_img[binary_img < threshold] = 0
    binary_img[binary_img >= threshold] = 1
    return binary_img


def myhist2(img, n_bins):
    H = np.zeros(n_bins)
    img_vector = img.reshape(-1)
    max_value = max(img_vector)
    min_value = min(img_vector)
    interval = (max_value - min_value) / n_bins
    for pixel in img_vector:
        idx = math.floor((pixel - min_value) / interval)
        if idx == n_bins:
            idx -= 1
        H[idx] += 1
    return H / sum(H)


def otsus_method(img, n_bins):
    histogram = myhist2(img, n_bins)  # get histogram
    within_class_variances = []
    for T in range(256):  # calculate for every possible threshold
        # calucalte weights for values < threshold
        weight_1 = sum(histogram[:T])
        # calucalte weights for values >= threshold
        weight_2 = sum(histogram[T:])
        # calucalte mean for values < threshold
        mean_1 = sum([i * histogram[i] for i in range(T)])
        # calucalte mean for values >= threshold
        mean_2 = sum([i * histogram[i] for i in range(T, n_bins)])
        # calculate variances for values < threshold
        var_1 = sum([(i - mean_1)**2 * histogram[i] for i in range(T)])
        # calculate variances for values >= threshold
        var_2 = sum([(i - mean_2)**2 * histogram[i] for i in range(T, n_bins)])
        within_class_variances.append(weight_1 * var_1 + weight_2 * var_2)

    # return the optimal threshold (minimum within class variance)
    return np.argmin(within_class_variances) / n_bins


def otsus_binary_mask(img):
    threshold = otsus_method(img, 256)
    binary_mask = create_binary_mask(img, threshold)
    return binary_mask

# (a) We will perform two basic morphological operations on the image mask.png, erosion and dilation.
# We will also experiment with combinations of both operations, named opening and closing.

# Question: Based on the results, which order of erosion and dilation operations produces opening and which closing?
# Opening: applying erosion, then dilation
# Closing: applying dilation, then erosion


def opening(mask, SE):  # apply erosion then dilation
    new_mask = np.copy(mask)
    new_mask = cv2.erode(new_mask, SE)
    new_mask = cv2.dilate(new_mask, SE)
    return new_mask


def closing(mask, SE):  # apply dilation then erosion
    new_mask = np.copy(mask)
    new_mask = cv2.dilate(new_mask, SE)
    new_mask = cv2.erode(new_mask, SE)
    return new_mask


mask_img = imread('images/mask.png')

n = 5
SE = np.ones((n, n))  # create a square structuring element
mask_after_closing = closing(mask_img, SE)
mask_closing_opening = opening(mask_after_closing, SE)
show_three_images(mask_img, "Original Mask", mask_after_closing,
                  "Closing, n = 5, Square SE", mask_closing_opening, "Closing then opening, n = 5, Square SE")

mask_after_opening = opening(mask_img, SE)
mask_opening_closing = closing(mask_after_opening, SE)
show_three_images(mask_img, "Original Mask", mask_after_opening, "Opening, n = 5, Square SE",
                  mask_opening_closing, "Opening then closing, n = 5, Square SE")

show_three_images(mask_img, "Original Mask", mask_after_closing,
                  "Closing, n = 5, Square SE", mask_after_opening, "Opening, n = 5, Square SE")

n = 2
SE = np.ones((n, n))  # create a square structuring element
mask_after_closing_2 = closing(mask_img, SE)
mask_closing_opening_2 = opening(mask_after_closing_2, SE)
show_three_images(mask_img, "Original Mask", mask_after_closing_2,
                  "Closing, n = 2, Square SE", mask_closing_opening_2, "Closing then opening, n = 2, Square SE")

mask_after_opening_2 = opening(mask_img, SE)
mask_opening_closing_2 = closing(mask_after_opening_2, SE)
show_three_images(mask_img, "Original Mask", mask_after_opening_2, "Opening, n = 2, Square SE",
                  mask_opening_closing_2, "Opening then closing, n = 2, Square SE")

show_three_images(mask_img, "Original Mask", mask_after_closing_2,
                  "Closing, n = 2, Square SE", mask_after_opening_2, "Opening, n = 2, Square SE")
n = 10
SE = np.ones((n, n))  # create a square structuring element
mask_after_closing_10 = closing(mask_img, SE)
mask_closing_opening_10 = opening(mask_after_closing_10, SE)
show_three_images(mask_img, "Original Mask", mask_after_closing_10,
                  "Closing, n = 10, Square SE", mask_closing_opening_10, "Closing then opening, n = 10, Square SE")

mask_after_opening_10 = opening(mask_img, SE)
mask_opening_closing_10 = closing(mask_after_opening_10, SE)
show_three_images(mask_img, "Original Mask", mask_after_opening_10, "Opening, n = 10, Square SE",
                  mask_opening_closing_10, "Opening then closing, n = 10, Square SE")

show_three_images(mask_img, "Original Mask", mask_after_closing_10,
                  "Closing, n = 10, Square SE", mask_after_opening_10, "Opening, n = 10, Square SE")

# (b) Try to clean up the mask of the image bird.jpg using morphological operations

bird_img = imread_gray('images/bird.jpg')
otsus_bird_mask = otsus_binary_mask(bird_img)

n = 25
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
cleaned_bird_mask = closing(otsus_bird_mask, SE)

show_three_images(bird_img, "Original Grayscale Image", otsus_bird_mask,
                  "Bird Mask using Otsu's Method", cleaned_bird_mask, "Closing, n = 25")

# (c) Write a function immask that accepts a three channel image and a binary mask and returns an image
# where pixel values are set to black if the corresponding pixel in the mask is equal to 0.


def immask(img, mask):
    new_mask = np.expand_dims(mask, axis=2)
    new_img = img * new_mask
    return new_img


orig_bird = imread('images/bird.jpg')
new_bird_img = immask(orig_bird, cleaned_bird_mask)
show_three_images(orig_bird, "Original Bird Image",
                  cleaned_bird_mask, "Binary Mask", new_bird_img, "immask")

# (d) Create a mask from the image in file eagle.jpg and visualize the result with immask

eagle_img = imread_gray('images/eagle.jpg')
inverted_eagle_img = 1 - eagle_img
eagle_mask = otsus_binary_mask(eagle_img)
inverted_eagle_mask = otsus_binary_mask(inverted_eagle_img)


size = 20
SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
cleaned_inverted_mask = closing(inverted_eagle_mask, SE2)
cleaned_mask = opening(eagle_mask, SE2)

orig_eagle = imread('images/eagle.jpg')
new_eagle_img_inverted = immask(orig_eagle, cleaned_inverted_mask)
new_eagle_img = immask(orig_eagle, cleaned_mask)

show_three_images(orig_eagle, "Original Eagle Image", cleaned_inverted_mask,
                  "Inverted Eagle Mask", new_eagle_img_inverted, "immask")
show_three_images(orig_eagle, "Original Eagle Image",
                  cleaned_mask, "Eagle Mask", new_eagle_img, "immask")

# (e) Write a script that loads the image coins.jpg, calculates a mask and cleans it up using morphological operations.
# Your goal is to get the coins as precisely as possible.
# Then, using connected components, remove the coins whose area is larger than 700 pixels from the original image

coins_img = 1 - imread_gray('images/coins.jpg')
coins_mask = otsus_binary_mask(coins_img)
cleaned_coins_mask = closing(coins_mask, SE2)
cleaned_coins_mask_uint = cleaned_coins_mask.astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    cleaned_coins_mask_uint, connectivity=8)
mask_to_remove = np.zeros_like(cleaned_coins_mask_uint)
for i in range(1, num_labels):  # Start from 1 to skip the background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area <= 700:
        mask_to_remove[labels == i] = 255

original_coins_img = imread('images/coins.jpg')
small_coins_img = np.copy(original_coins_img)
small_coins_img[mask_to_remove == 0] = 1
show_three_images(original_coins_img, "Original Coins Image",
                  mask_to_remove, "Mask", small_coins_img, "After removing big coins")
