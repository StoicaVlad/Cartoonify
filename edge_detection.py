from convolution import *
import matplotlib.pyplot as plt


weak_pixel = 55
strong_pixel = 255
lowthreshold = 0.05
highthreshold = 0.05


# apply grayscale
def grayscale(image):
    color_eye_values = [0.299, 0.587, 0.114]
    red_pixel, green_pixel, blue_pixel = image[..., 0], image[..., 1], image[..., 2]
    gray_img = color_eye_values[0] * red_pixel + \
               color_eye_values[1] * green_pixel + \
               color_eye_values[2] * blue_pixel
    return gray_img


# apply gaussian blur
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


# apply sobel filter
def sobel_filters(img):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], float)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], float)

    Ix = apply_convolution(img, Kx)
    Iy = apply_convolution(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    no_rows, no_columns = img.shape
    Z = np.zeros((no_rows, no_columns), dtype=int)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, no_rows - 1):
        for j in range(1, no_columns - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z


def threshold(image):
    high_threshold = image.max() * highthreshold
    low_threshold = high_threshold * lowthreshold

    no_rows, no_columns = image.shape
    thresholded_image = np.zeros((no_rows, no_columns), dtype=int)

    strong_i, strong_j = np.where(image >= high_threshold)
    _, zeros_j = np.where(image < low_threshold)

    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    thresholded_image[strong_i, strong_j] = strong_pixel
    thresholded_image[weak_i, weak_j] = weak_pixel

    return thresholded_image


def hysteresis(image):
    no_rows, no_columns = image.shape

    for i in range(1, no_rows - 1):
        for j in range(1, no_columns - 1):
            if image[i, j] == weak_pixel:
                try:
                    if ((image[i + 1, j - 1] == strong_pixel)
                            or (image[i + 1, j] == strong_pixel)
                            or (image[i + 1, j + 1] == strong_pixel)
                            or (image[i, j - 1] == strong_pixel)
                            or (image[i, j + 1] == strong_pixel)
                            or (image[i - 1, j - 1] == strong_pixel)
                            or (image[i - 1, j] == strong_pixel)
                            or (image[i - 1, j + 1] == strong_pixel)):
                        image[i, j] = strong_pixel
                    else:
                        image[i, j] = 0
                except IndexError:
                    pass
    return image


def apply_canny_filter(input_image):
    gray_image = grayscale(input_image)
    img_smoothed = apply_convolution(gray_image, gaussian_kernel(2, 2))
    gradient_mat, theta_mat = sobel_filters(img_smoothed)
    non_max_img = non_max_suppression(gradient_mat, theta_mat)
    threshold_img = threshold(non_max_img)
    final_img = hysteresis(threshold_img)
    return final_img
