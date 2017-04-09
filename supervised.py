#!/usr/bin/env python3

# Test for supervised machine learning
# "A computer program is said to learn from experience E with respect to some
# class of tasks T and performance measure P if its performance at tasks in T,
# as measured by P, improves with experience E."

import random

class supervised_environment():
    def __init__(self, examples=10):
        self.data = []
        self.init_data(examples)
#        self.find_delta()

    def init_data(self, examples):
        for round in range(examples):
            value_1 = random.randint(0, 10)
            value_2 = random.randint(0, 10)
            self.data.append((min(value_1, value_2), max(value_1, value_2)))

    def calc_result(self):
        diff = 0
        for round in range(len(self.data)):
            diff += (self.data[round][1] - self.data[round][0])
        return diff / len(self.data)








m = supervised_environment()
print(m.data)
print(m.calc_result())
