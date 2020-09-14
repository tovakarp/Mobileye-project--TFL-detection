import numpy as np
import scipy.ndimage as ndimage


def convolve(img):
    kernel = np.ones((3, 3)) / -9
    kernel[1, 1] = 8 / 9
    return ndimage.convolve(img, kernel)


def get_light_positions(c_image):
    convolved_img = convolve(c_image)
    filtered_img = ndimage.maximum_filter(convolved_img, 10)
    green_position = np.argwhere(filtered_img == convolved_img)
    green_position = list(filter(lambda l: convolved_img[l[0]][l[1]] > 0.1, green_position))
    x_pos = [p[0] for p in green_position]
    y_pos = [p[1] for p in green_position]

    return y_pos, x_pos


def find_lights(c_image):
    x_red, y_red = get_light_positions(c_image[:, :, 0])
    x_green, y_green = get_light_positions(c_image[:, :, 1])

    light_candidtes = []
    candidate_color = []

    for x, y in zip(x_red, y_red):
        light_candidtes.append([x, y])
        candidate_color.append('r')

    for x, y in zip(x_green, y_green):
        light_candidtes.append([x, y])
        candidate_color.append('g')

    return light_candidtes, candidate_color


def visualize(image, candidates, colors, fig, title):
    fig.set_title(title)
    fig.imshow(image)
    fig.set_xticks([])
    fig.set_yticks([])

    x_red = [x[0] for i, x in enumerate(candidates) if colors[i] == 'r']
    y_red = [y[1] for i, y in enumerate(candidates) if colors[i] == 'r']

    x_green = [x[0] for i, x in enumerate(candidates) if colors[i] == 'g']
    y_green = [y[1] for i, y in enumerate(candidates) if colors[i] == 'g']

    fig.plot(x_red, y_red, 'r.')
    fig.plot(x_green, y_green, 'g.')