import numpy as np

class dataset:

    def __init__(self, pirna, mrna, label=None, mode='train'):
        self.pirna = pirna
        self.mrna = mrna
        self.label = label
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.pirna[item], self.mrna[item], self.label[item]

        return self.pirna[item], self.mrna[item]

    def __len__(self):
        return len(self.pirna)

