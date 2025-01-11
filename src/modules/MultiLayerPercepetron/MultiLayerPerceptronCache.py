
class MultiLayerPerceptronCache:
    def __init__(self):
        self.errorByEpoch = []
        self.errorLossBySample=[]
        self.errorsBySample=[]
        self.regBySample = []
        self.weightsByEpoch = []
        self.biasByEpoch = []