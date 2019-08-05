import numpy as np
import cv2
from skimage.color import rgb2gray
import math
from utils.gf import guided_filter
import matplotlib.pyplot as plt
import pdb

def brightest_pixels_count(num_pixels, fraction):
    tmp = math.floor(fraction*num_pixels)
    return tmp+((tmp+1) % 2)


def clip_to_unit_range(image):
    clipped_image = np.maximum(np.minimum(image, 1), 0)
    return clipped_image


def get_dark_channel(image, window):
    # erode the image
    kernel = np.ones((window, window), np.uint8)
    image_erode = cv2.erode(image, kernel, iterations=1)
    dark_channel = np.min(image_erode, axis=-1)
    return image_erode, dark_channel


def get_atmosphere_light(dark_channel, image):

    # Determine the number of brightest pixels in dark channel
    brightest_pixels_frac = 1e-3
    H, W = dark_channel.shape
    num_pixels = H*W
    brightest_pixels_num = brightest_pixels_count(num_pixels, brightest_pixels_frac)

    # get the indices of brightest pixels in dark channel
    sort_idx = np.argsort(np.ndarray.flatten(dark_channel))[::-1]
    brightest_pixels_idx = sort_idx[:brightest_pixels_num]

    gray_image = rgb2gray(image)
    gray_brightest_pixels = np.ndarray.flatten(gray_image)[brightest_pixels_idx]

    gray_median_intensity = np.median(gray_brightest_pixels)
    temp_idx = np.where(gray_brightest_pixels==gray_median_intensity)[0][0]
    x, y = np.unravel_index(brightest_pixels_idx[temp_idx], (H, W))

    L = image[x, y, :]
    return L, brightest_pixels_idx[temp_idx]


def init_transmission(I_eroded, L):
    omega = 0.95
    H, W, _ = I_eroded.shape
    L_expand = np.repeat(np.repeat(L[np.newaxis, :], W, axis=0)[np.newaxis, :], H, axis=0)
    t = 1 - omega*np.min(I_eroded/L_expand, axis=-1)
    return t


def inverse_model(I, t, A, t_thresh):
    """
    :param I: hazy image of shape (H, W, C)
    :param t: estimated transmission map of shape (H, W)
    :param A: estimated atmosphere light of shape (C, )
    :param t_thresh: threshold for transmission map
    :return: dehazed image using I = J(x)t(x)+A(1-t(x))
    """
    t_expand = np.repeat(t[...,np.newaxis], 3, axis=-1)
    J = (I-A[np.newaxis, np.newaxis, :]*(1-t_expand))/t_expand
    return J


def dark_channel_dehaze(I):
    window_size = 15
    t_threshold = 0.1
    window_size_guided_filter = 41
    epsilon = 1e-3

    I_erode, I_dark = get_dark_channel(I, window_size)
    L, _= get_atmosphere_light(I_dark, I)
    t_initial = init_transmission(I_erode, L)
    t = guided_filter(I, t_initial, window_size_guided_filter, epsilon)
    t_clip = clip_to_unit_range(t)
    J = inverse_model(I, t_clip, L, t_threshold)
    return clip_to_unit_range(J)

if __name__ == '__main__':
    hazy_image = plt.imread('../01_00002_rgb.tiff')
    hazy_image = hazy_image/255.0
    clean_image = dark_channel_dehaze(hazy_image)