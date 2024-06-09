import matplotlib.pyplot as plt

import edge_detection
import color_reduction

stroke_color = [0, 0, 0]
input_data = [
    #r"input_data/input1.png",
    #r"input_data/input2.png",
    r"input_data/input_3.jpg",
    #r"input_data/input_4.jpg"
    ]


def print_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


if __name__ == '__main__':

    for image_path in input_data:
        sample_img = plt.imread(image_path)
        print_image(sample_img)
        edges = edge_detection.apply_canny_filter(sample_img)
        #print_image(edges)
        reduced_image = color_reduction.apply_color_reduction(sample_img, 4)
        #print_image(reduced_image)

        for i in range(len(edges)):
            for j in range(len(edges[i])):
                if edges[i][j] != 0:
                    reduced_image[i][j] = stroke_color

        print_image(reduced_image)
