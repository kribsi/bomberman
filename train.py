from collections import namedtuple, deque
import copy
import math
import numpy as np
import os
import pickle
from typing import List
import events as e

MY_ACTIONS = ["coin", "bomb opponent", "bomb crate", "find action"]

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.alpha = 0.5
    self.gamma = 0.5
    QTABLE = {}
    V = {}
    if os.path.isfile("var.pickle"):
        with open("my-saved-model.pt", "rb") as file:
            QTABLE, V = pickle.load(file)
    else:
        states = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for m in range(4):
                            state=calculate_state_number([i,j,k,l,m])
                            states.append(state)
                            # initial guess:
                            QTABLE[(state, 'coin')]=-j
                            QTABLE[(state, 'bomb opponent')]=k
                            QTABLE[(state, 'bomb crate')]=i+l
                            QTABLE[(state, 'find action')]=-m
                            V[state] = max(-j,k,i+l,-m)
        to_save = (QTABLE, V)
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(to_save, file)
    self.Q,self.V = copy.deepcopy(QTABLE), copy.deepcopy(V)
    self.Q_prev, self.V_prev=copy.deepcopy(self.Q), copy.deepcopy(self.V)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.state=calculate_state_number(self.features)
    if (self.prev_features ==()) : return
    self.prev_state=calculate_state_number(self.prev_features)
    reward=reward_from_events(self,events)

    self.transitions.append(Transition(self.prev_state, self.my_action, self.state, reward))
 
    nsteplearn(self,reward)
    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.state=calculate_state_number(self.features)
    self.prev_state=calculate_state_number(self.prev_features)
    reward=reward_from_events(self,events)
    self.transitions.append(Transition(self.prev_state, self.my_action, self.state, reward))
 
    nsteplearn(self, reward)

    QTABLE=self.Q
    V=self.V

    to_save = (QTABLE, V)
    with open("my-saved-model.pt", "wb") as file:
       pickle.dump(to_save, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.GOT_KILLED: -30,
        e.KILLED_SELF: -30,
        e.CRATE_DESTROYED : 1,
        e.COIN_FOUND : 3,
        e.BOMB_DROPPED :1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
TRANSITION_HISTORY_SIZE = 10  

def calculate_state_number(array) :
    i,j,k,l,m= array
    return i*10000+j*1000+k*100+l*10+m

def nsteplearn (self,reward):
        if (len(self.transitions)<TRANSITION_HISTORY_SIZE) : return
        sum=0
        state = None
        action = None
        for i in range(len(self.transitions)):
            state, action, next_state, reward = self.transitions[i]
            sum+= (math.pow(self.gamma, (i))*reward)
        ago_state, ago_action =state, action
        self.Q[(ago_state, ago_action)]=self.Q_prev[(ago_state,ago_action)]+self.alpha*(sum+math.pow((self.gamma),TRANSITION_HISTORY_SIZE)*self.V_prev[state]-self.Q_prev[(ago_state, ago_action)])
        list=[]
        for action in MY_ACTIONS :
            list.append(self.Q[(ago_state, action)])
        print(list)
        self.V[ago_state] = max(list) 
        self.transitions.popleft()
        return 