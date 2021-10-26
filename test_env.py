import numpy as np
import random

def check_if_end(state):  # turn: 1 or 2, 5개씩
    count = 0
    for i in range(0, 3):
        if state[i] == 1:
            count += 1
    if count == 3: return True
    else: return False

class TestEnv():

    reward = 0

    def __init__(self):
        self.row_number = 3
        self.state = np.zeros(self.row_number)

    def step(self, action): # action: 0~2

        done = False
        reward = 0

        if self.state[action] == 0:
            self.state[action] = 1
            reward = 0
            if check_if_end(self.state):
                done = True
                reward = 1
                return self.state, reward, done
            else:
                done = False
                return self.state, reward, done
        else:   # illegal
            done = True
            reward = - 1
            return self.state, reward, done

    def reset(self):
        self.row_number = 3
        self.state = np.zeros(self.row_number)
        return self.state
