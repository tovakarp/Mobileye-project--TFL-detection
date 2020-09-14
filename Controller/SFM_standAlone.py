import numpy as np
import matplotlib._png as png

from Controller import SFM


def visualize(prev_container, curr_container, focal, pp, fig):
    if not len(curr_container.traffic_light):
        return

    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

    fig.set_title('TFL Distances')
    fig.imshow(curr_container.img)
    curr_p = curr_container.traffic_light
    fig.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

    for i in range(len(curr_p)):
        fig.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if curr_container.valid[i]:
            fig.text(curr_p[i, 0], curr_p[i, 1], r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]),
                     color='r', fontsize=6)
    fig.plot(foe[0], foe[1], 'r+')
    fig.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')

    fig.set_xticks([])
    fig.set_yticks([])


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = png.read_png_int(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []

        self.img_path = img_path
        self.auxiliary = []
