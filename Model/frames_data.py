class FramesData:
    @staticmethod
    def calc_EM(cur_id, pkl_data):
        return pkl_data['egomotion_' + str(cur_id - 1) + '-' + str(cur_id)]

    def __init__(self, pkl_data):
        self.focal = pkl_data['flx']
        self.pp = pkl_data['principle_point']
        self.EMs = [self.calc_EM(i + 24, pkl_data) for i in range(1,6)]
