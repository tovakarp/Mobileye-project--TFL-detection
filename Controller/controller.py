import pickle

from Controller.TflManager import TflManager


class Controller:
    def __init__(self, file_path: str) -> None:
        pls_data = self.read_pls(file_path)
        self.frames = pls_data[1:]
        self.pkl_data = self.read_pkl(pls_data[0])
        self.tfl_manager = TflManager(self.pkl_data)

    @staticmethod
    def read_pls(file_path):
        with open(file_path) as the_file:
            return the_file.read().split("\n")

    @staticmethod
    def read_pkl(file_path):
        with open(file_path, 'rb') as pklfile:
            return pickle.load(pklfile, encoding='latin1')

    def run(self):
        prev = None

        for index in range(len(self.frames)):
            prev = self.tfl_manager.run(self.frames[index], prev, index)
