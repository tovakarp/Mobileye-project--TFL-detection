import matplotlib.pyplot as plt


class Output:
    def __init__(self):
        self.fig, (self.light_src, self.tfl, self.distances) = plt.subplots(3, 1, figsize=(20, 10))

    def show(self):
        plt.show()

