import numpy as np
import random

class Game(object):
    def __init__(self, f, c, s, v, sender_type, data, debug=False):
        self.f = f
        self.c = c
        self.s = s
        self.v = v
        self.target = None
        self.states = []
        self.sender_type = sender_type
        self.debug=debug
        self.X1 = data[0]
        self.Y1=data[1]
        self.reset()

    def reset(self):
        self.nos = np.random.choice(range(self.c), self.s, replace=False).tolist()
        self.states = [self.X1[self.Y1==i][np.random.choice(self.X1[self.Y1==i].shape[0], 1)][0] for i in self.nos]
        self.target = 0

    def speaker_input(self, img):
        if self.sender_type == "aware":
            inp = list(np.array(self.states).flatten())
        elif self.sender_type=="agnostic":
            inp = list(self.states[0])
        else:
            print("Invalid sender type")
            return
        inp.extend(list(img.flatten()))
        return inp

    def listener_input(self, features):
        shuffled_index, shuffled_states = zip(*sorted(zip(range(self.s), self.states), key=lambda _: random.random()))
        self.target = list(shuffled_index).index(0)
        lst = np.array(shuffled_states).flatten().tolist()
        lst.extend(features)
        if self.debug:
            print(lst)
        return lst

    def reward(self, out):
        assert self.target is not None
        if out == self.target:
            return 1
        else:
            return 0
