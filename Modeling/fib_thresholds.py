#! python3
# fib_thresholds.py
# RTS - January 2019

# Class that keeps track of FIB California state standards (current as of January 2019)
# Initialize in every script to automatically have the thresholds

class FIB():

    def __init__(self):
        self.fib_list = ['TC', 'FC', 'ENT']
        self.fib_thresholds = {
            'TC': 10000,
            'FC': 400,
            'ENT': 104
            }
