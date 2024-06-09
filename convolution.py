import numpy as np


def convolution_calculate_target_size(img_size: int, kernel_size: int):
    no_pixels = 0

    # From 0 up to img size
    for i in range(img_size):
        added = i + kernel_size
        # It must be lower than the image size
        if added <= img_size:
            no_pixels += 1

    return no_pixels


def apply_convolution(image, kernel):
    tgt_size_x = convolution_calculate_target_size(
        img_size=image.shape[0],
        kernel_size=kernel.shape[0]
    )
    tgt_size_y = convolution_calculate_target_size(
        img_size=image.shape[1],
        kernel_size=kernel.shape[0]
    )
    k = kernel.shape[0]

    convolved_img = np.zeros(shape=(tgt_size_x, tgt_size_y))

    for i in range(tgt_size_x):
        for j in range(tgt_size_y):
            mat = image[i: i+k, j: j+k]
            # Apply the convolution - element-wise multiplication and summation of the result
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img
