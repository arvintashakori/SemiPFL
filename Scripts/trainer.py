import numpy as np
import pandas as pd

class params:
    def __init__(self):
        self.max_number_of_users = 59
        self.max_number_of_trials = 6
        self.window_size = 30
        self.input_dim = 9

def data_loader (user_id, trial_id):
    par = params()
    if user_id > par.max_number_of_users or user_id < 0:
        print("Please enter a valid user id")
        return
    if trial_id > par.max_number_of_trials or trial_id < 0:
        print("Please enter a valid trial id")
    return np.load('Mobiact/user' + str(user_id - 1) + 'trail_' + str(trial_id - 1) + '.npy')


data_loader(user_id = 10, trial_id = 4)
