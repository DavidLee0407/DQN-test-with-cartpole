import numpy as np
import random

def check_if_end(state, turn):  # turn: 1 or 2, 5개씩

    count = 0

    for j in range(0, 3):
        for i in range(0, 3):
            if state[i][j] == turn:
                count += 1
        if count==3: return True
        else: count = 0

    for j in range(0, 3):
        for i in range(0, 3):
            if state[j][i] == turn:
                count += 1
        if count==3: return True
        else: count = 0

    if (state[0][0]==turn) and (state[1][1]==turn) and (state[2][2]==turn):
        return True
    elif (state[2][0]==turn) and (state[1][1]==turn) and (state[0][2]==turn):
        return True

class TicTacToeEnv:

    reward = 0

    def init(self):
        self.rows = 3
        self.columns = 3
        self.state = np.zeros((self.rows, self.columns))
        self.turn = 2   #1: O 차례, 2: X 차례
        self.available_spaces = self.rows*self.columns
        self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def step(self, action):

        done = False

        if self.turn==1: self.turn==2
        elif self.turn==2: self.turn==1

        reward = 0

        self.action = action   # index, 0~8
        try: self.action_list.remove(action)
        except ValueError:
            pass
        action_index_1 = action % 3
        action_index_2 = action//3

        if self.state[action_index_2][action_index_1]==0:
            self.state[action_index_2][action_index_1] = 1
        else:
            self.state[action_index_2][action_index_1] = 1
            reward = - 10
            done = True

        if done: pass
        else:
            if (check_if_end(self.state, 1) == True):
                done = True
                reward = 1
            else:
                done = False

            if done: pass
            else:
                if len(self.action_list) > 0:
                    a = random.randint(0, len(self.action_list)-1)
                    random_action = self.action_list[a]
                    action_index_1 = random_action % 3
                    action_index_2 = random_action//3
                    self.state[action_index_2][action_index_1] = 2
                    self.action_list.remove(random_action)   

                    if (check_if_end(self.state, 2) == True):
                        done = True
                        reward = -1
                    else:
                        done = False
                else:
                    done = True
                    pass

        info = {}
        return self.state, reward, done, info


    def render(self):
        #self.state[self.action[2]-1][self.action[1]-1] = self.turn
        #print(self.turn)
        pass


    def reset(self):
        self.state = np.zeros((3, 3))
        self.turn = 1   
        self.available_spaces = 3*3
        self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        return self.state