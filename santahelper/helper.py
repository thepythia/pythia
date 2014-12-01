__author__ = 'phoenix'

import csv
from datetime import datetime

class Helper:

    def __init__(self):
        self.elvesCount = 1325
        self.elvesProd = [1.0 for i in range(self.elvesCount)]
        self.elvesState = [0 for i in range(self.elvesCount)] #0 idle 1 working

    def run(self, filename):
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                print(row)



    def scheduling(self, order):
        assert len(order) == 3
        toyId = order[0]
        te = [int(i) for i in order[1].split(' ')]
        startTime = datetime(te[0], te[1], te[2], te[3], te[4])
        duration = order[2] #in minutes


