import numpy as np
import time, os
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, learning_params, testing_params, min_steps = 1000, total_steps = 10000):
        """
        Parameters
        ----------
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        testing_params : TestingParameters object
            Object storing parameters to be used in testing.
        min_steps : int
        total_steps : int
            Total steps allowed before stopping learning.
        """
        self.learning_params = learning_params
        self.testing_params = testing_params

        # Keep track of the number of learning/testing steps taken
        self.min_steps = min_steps
        self.total_steps = total_steps
        self.current_step = 0

        # Store the results here
        self.results = {}
        self.steps = []

    # Methods to keep track of trainint/testing progress
    def restart(self):
        self.current_step = 0

    def add_step(self):
        self.current_step += 1

    def get_current_step(self):
        return self.current_step

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def stop_task(self, step):
        return self.min_steps <= step