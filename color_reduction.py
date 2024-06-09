import numpy as np
import copy


def median_cut_quantize(img, img_arr):
    # make color quantity
    red_avg = np.mean(img_arr[:, 0])
    green_avg = np.mean(img_arr[:, 1])
    blue_avg = np.mean(img_arr[:, 2])

    for data in img_arr:
        img[int(data[3])][int(data[4])] = [red_avg, green_avg, blue_avg]


def fragment_zone(img, img_arr, depth):
    if len(img_arr) == 0:
        return

    if depth == 0:
        median_cut_quantize(img, img_arr)
        return

    red_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    green_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    blue_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

    space_with_highest_range = 0

    if green_range >= red_range and green_range >= blue_range:
        space_with_highest_range = 1
    elif blue_range >= red_range and blue_range >= green_range:
        space_with_highest_range = 2
    elif red_range >= blue_range and red_range >= green_range:
        space_with_highest_range = 0

    # sort by most common color and calculate the median
    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)

    # split the array into two smaller zones along the median
    fragment_zone(img, img_arr[0:median_index], depth - 1)
    fragment_zone(img, img_arr[median_index:], depth - 1)


# make image flat
def flatten_image(sample_img):
    flattened_img_array = []
    for row_index, rows in enumerate(sample_img):
        for column_index, color in enumerate(rows):
            flattened_img_array.append([color[0], color[1], color[2], row_index, column_index])

    return np.array(flattened_img_array)


def apply_color_reduction(input_img, no_colors):
    test_img = copy.deepcopy(input_img)
    flattened_img_array = flatten_image(input_img)
    fragment_zone(test_img, flattened_img_array, no_colors)

    return test_img
