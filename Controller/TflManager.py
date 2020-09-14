import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from Controller import detect_light, SFM_standAlone, predict_tfl, SFM
from Model.frames_data import FramesData
from View.output import Output


class TflManager:
    def __init__(self, pkl_data):
        self.model = load_model(r"C:\Users\RENT\Desktop\Detect_TFLs\Model\model.h5")
        self.frames_data = FramesData(pkl_data)

    def detect_lights(self, img, fig):
        print("Detecting lights")
        img = plt.imread(img).copy()
        candidates, auxiliary = detect_light.find_lights(img)
        detect_light.visualize(img, candidates, auxiliary, fig, "Light Detection")

        return candidates, auxiliary

    def predict_tfls(self, img, light_candidates, light_auxiliary, fig):
        print("Predict TFL")

        img = plt.imread(img).copy()
        images, candidates, auxiliary = predict_tfl.crop_all_images(img, light_candidates, light_auxiliary)
        images *= 255

        predictions = self.model.predict(images)

        assert len(predictions) <= len(light_candidates)

        light_candidates = [candidate for index, candidate in enumerate(candidates) if predictions[index, 1] > 0.9995]
        light_auxiliary = [candidate for index, candidate in enumerate(auxiliary) if predictions[index, 1] > 0.9995]

        detect_light.visualize(img, light_candidates, light_auxiliary, fig, "TFL Prediction")

        return light_candidates, light_auxiliary

    def calc_distances(self, curr_container, prev_container, fig):
        print("Calculating SFM")
        curr_container.traffic_light = np.array(curr_container.traffic_light)
        prev_container.traffic_light = np.array(prev_container.traffic_light)
        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.frames_data.focal, self.frames_data.pp)
        SFM_standAlone.visualize(prev_container, curr_container, self.frames_data.focal, self.frames_data.pp, fig)

    def run(self, frame, prev_container, index):
        print(f"*** {index + 1} ***")

        output = Output()

        curr_container = SFM_standAlone.FrameContainer(frame)

        curr_container.traffic_light, curr_container.auxiliary = self.detect_lights(curr_container.img_path, output.light_src)

        curr_container.traffic_light, curr_container.auxiliary = self.predict_tfls(curr_container.img_path, curr_container.traffic_light,
                                                                  curr_container.auxiliary, output.tfl)

        if index:
            curr_container.EM = self.frames_data.EMs[index - 1]
            self.calc_distances(curr_container, prev_container, output.distances)

        output.show()

        return curr_container
