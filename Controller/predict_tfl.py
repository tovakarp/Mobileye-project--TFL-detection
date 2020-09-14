import numpy as np


def crop(x, y, img):
    return img[x - 81 // 2: x + 81 // 2 + 1, y - 81 // 2: y + 81 // 2 + 1, :]


def crop_all_images(img, candidates, colors):
    croped_imgs = []
    can = []
    color = []
    for i, c in enumerate(candidates):
        croped = crop(c[1], c[0], img)
        if croped.shape == (81, 81, 3):
            croped_imgs.append(croped)
            can.append(c)
            color.append(colors[i])

    return np.array(croped_imgs), can, color
