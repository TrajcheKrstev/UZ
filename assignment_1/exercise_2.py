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


def show_hist(hist, n_bins):
    edges = np.arange(n_bins)
    plt.bar(edges, hist)
    plt.title(str(n_bins) + " bins")
    plt.show()


def show_two_histograms(hist1, title1, hist2, title2):
    plt.subplot(1, 2, 1)
    edges = np.arange(len(hist1))
    plt.bar(edges, hist1)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    edges = np.arange(len(hist2))
    plt.bar(edges, hist2)
    plt.title(title2)
    plt.show()


def show_three_histograms(hist1, title1, hist2, title2, hist3, title3):
    plt.subplot(1, 3, 1)
    edges = np.arange(len(hist1))
    plt.bar(edges, hist1)
    plt.title(title1)
    plt.subplot(1, 3, 2)
    edges = np.arange(len(hist2))
    plt.bar(edges, hist2)
    plt.title(title2)
    plt.subplot(1, 3, 3)
    edges = np.arange(len(hist3))
    plt.bar(edges, hist3)
    plt.title(title3)
    plt.show()


def show_img_and_two_histograms(img, img_title, hist1, title1, hist2, title2):
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.title(img_title)
    plt.subplot(1, 3, 2)
    edges = np.arange(len(hist1))
    plt.bar(edges, hist1)
    plt.title(title1)
    plt.subplot(1, 3, 3)
    edges = np.arange(len(hist2))
    plt.bar(edges, hist2)
    plt.title(title2)
    plt.show()

# (a) Create a binary mask from a grayscale image.


grayscale_img = imread_gray('images/bird.jpg')


def create_binary_mask_1(img, threshold):
    binary_img = np.copy(img)
    binary_img[binary_img < threshold] = 0
    binary_img[binary_img >= threshold] = 1
    return binary_img


def create_binary_mask_2(img, threshold):
    binary_img = np.copy(img)
    binary_img = np.where(binary_img < threshold, 0, 1)
    return binary_img


binary_mask_1 = create_binary_mask_1(grayscale_img, 0.3)
binary_mask_2 = create_binary_mask_2(grayscale_img, 0.3)

show_two_images(grayscale_img, "Original Grayscale Image", binary_mask_1,
                "Binary Mask")

# ------------------------------------------------------------------------------------------------

# (b) Write a function myhist that accepts a grayscale image and the number of bins that will be used in building a histogram.


def myhist(img, n_bins):
    H = np.zeros(n_bins)
    img_vector = img.reshape(-1)
    for pixel in img_vector:
        idx = math.floor(pixel * n_bins)
        if idx == n_bins:
            idx -= 1
        H[idx] += 1
    return H / sum(H)  # we divide by sum(H) to normalize the values


histogram_1 = myhist(grayscale_img, 20)
histogram_2 = myhist(grayscale_img, 100)
show_img_and_two_histograms(
    grayscale_img, "Grayscale Image", histogram_1, "20 Bins", histogram_2, "100 bins")

# Question: The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?
# Normalized histograms represent the probability density function of pixel intensities.
# Each bin value in the normalized histogram indicates the probability of finding a pixel with the corresponding intensity in the image.

# (c) Modify your function myhist to no longer assume the [0,1] range for values.
# Instead, it should find the maximum and minimum values in the image and calculate the bin ranges based on these values.
# Write a script that shows the difference between both versions of the function.


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


myhist2_histogram = myhist2(grayscale_img, 20)
show_img_and_two_histograms(grayscale_img, "Grayscale Image", histogram_1,
                            "myhist 20 Bins", myhist2_histogram, "myhist2 20 Bins")


# (d) Test myhist function on images (three or more) of the same scene in different lighting conditions.

first_img = imread_gray("images/img_1.jpg")
second_img = imread_gray("images/img_2.jpg")
third_img = imread_gray("images/img_3.jpg")

first_img_hist_1 = myhist2(first_img, 20)
first_img_hist_2 = myhist2(first_img, 100)
show_img_and_two_histograms(first_img, "First Grayscale Image",
                            first_img_hist_1, "20 Bins", first_img_hist_2, "100")
second_img_hist_1 = myhist2(second_img, 20)
second_img_hist_2 = myhist2(second_img, 100)
show_img_and_two_histograms(second_img, "Second Grayscale Image",
                            second_img_hist_1, "20 Bins", second_img_hist_2, "100")

third_img_hist_1 = myhist2(third_img, 20)
third_img_hist_2 = myhist2(third_img, 100)
show_img_and_two_histograms(third_img, "Third Grayscale Image",
                            third_img_hist_1, "20 Bins", third_img_hist_2, "100")


show_three_histograms(first_img_hist_1, "First Image 20 Bins", second_img_hist_1,
                      "Second Image 20 Bins", third_img_hist_1, "Third Image 20 Bins")


show_three_histograms(first_img_hist_2, "First Image 100 Bins", second_img_hist_2,
                      "Second Image 100 Bins", third_img_hist_2, "Third Image 100 Bins")


# (e) Implement Otsu’s method for automatic threshold calculation. It should accept a grayscale image and return the optimal threshold.
# Using normalized histograms, the probabilities of both classes are easy to calculate.
# Write a script that shows the algorithm’s results on different images.


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
    binary_mask = create_binary_mask_1(img, threshold)
    return binary_mask

# Thresholding on bird.jpg


otsus_bird_mask = otsus_binary_mask(grayscale_img)
show_two_images(grayscale_img, "Grayscale Bird Image",
                otsus_bird_mask, "Binary Mask using Otsu's Method")

# Thresholding on eagle.jpg

eagle_img = imread_gray('images/eagle.jpg')
inverted_eagle_img = 1 - eagle_img
otsus_eagle_mask = otsus_binary_mask(inverted_eagle_img)
show_two_images(eagle_img, "Grayscale Eagle Image",
                otsus_eagle_mask, "Binary Mask using Otsu's Method")

# Thresholding on candy.jpg

candy_img = imread_gray('images/candy.jpg')
inverted_candy_img = 1 - candy_img
otsus_candy_mask = otsus_binary_mask(inverted_candy_img)
show_two_images(candy_img, "Grayscale Candy Image",
                otsus_candy_mask, "Binary Mask using Otsu's Method")
