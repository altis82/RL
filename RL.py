# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 08:52:02 2022

@author: Chuan Pham
"""

import random
from collections import defaultdict, namedtuple
from itertools import product, starmap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Image, YouTubeVideo
from scipy import stats

#sns.set()
# states_colors = matplotlib.colors.ListedColormap(
#     ['#9A9A9A', '#D886BA', '#4D314A', '#6E9183'])
# cmap_default = 'Blues'
# cpal_default = sns.color_palette(("Blues_d"))

# sns.set_style("white")
# sns.set_context("poster")
# random.seed(1)


#define the environment based on namedtuple
# the following uses the namedtuple function to create the State class:
State = namedtuple('State', ['row', 'col','con'])#row: row, col: col, con: condition
all_states =[]
rows =4
cols=12

for i in range(rows):
    for j in range(cols):
        if i==0 and j ==0:
            #start state
            all_states.append(State(i,j,2))
        elif i==0 and j==11:
            #destination state
            all_states.append(State(i,j,1))
        elif i==0 and j%2==0:
            #cliff states
            all_states.append(State(i,j,-1))
        else:
            all_states.append(State(i,j,0))

            
#we define an array that contains all states
# all_states = [State(0, 0), State(0, 1), State(0, 2), State(0, 3), State(0, 4),
#               State(0, 5), State(0, 6), State(0, 7), State(0, 8), State(0, 9),
#               State(0, 10), State(0, 11), State(1, 0), State(1, 1),
#               State(1, 2), State(1, 3), State(1, 4), State(1, 5), State(1, 6),
#               State(1, 7), State(1, 8), State(1, 9), State(1, 10),
#               State(1, 11), State(2, 0), State(2, 1), State(2, 2), State(2, 3),
#               State(2, 4), State(2, 5), State(2, 6), State(2, 7), State(2, 8),
#               State(2, 9), State(2, 10), State(2, 11), State(3, 0),
#               State(3, 1), State(3, 2), State(3, 3), State(3, 4), State(3, 5),
#               State(3, 6), State(3, 7), State(3, 8), State(3, 9), State(3, 10),
#               State(3, 11)]

#print(all_states)

states_colors = matplotlib.colors.ListedColormap(
    ['#9A9A9A', '#D886BA', '#4D314A', '#6E9183'])
data=np.zeros((rows,cols))
annotation_movement=np.full((4,12), -1, dtype='str')
def draw_map(all_states):
    
    for state in all_states:
        #start 
        if state.con==2:
            data[state.row,state.col]=2
        #destination
        elif state.con==1:
            data[state.row, state.col]=1
        #cliff
        elif state.con==-1:
            data[state.row, state.col]=-1
        else:
            data[state.row, state.col]=0
            
    sns.heatmap(data,cmap=states_colors,  cbar=False, square=True, linewidths=1, fmt='')
    plt.show()

draw_map(all_states)

# cliff_states = all_states[1:11]
# goal_state = State(m=0, n=11)
# start_state = State(m=0, n=0)

# terminal = cliff_states + [goal_state]

dflt_reward = -1
cliff_reward = -100
#(row,col)
steps={'<':(0,-1),
           '>':(0,1),
           '^':(-1,0),
           'v':(1,0)}

#

def get_state(state_location):
    """
    location is a tuple (row,col)

    Parameters
    ----------
    location : TYPE
        DESCRIPTION.

    Returns
    State
    -------
    None.

    """
    state=all_states[0]
    for state in all_states:
        if state.row==state_location[0] and state.col==state_location[1]:
            return state
    return state
#we define a class CliffWorld


class CliffWorld:
    
    def __init__(self, start_state):
        self.record_list=[]
        self.start_state= start_state
        self.log_dict={}
        self.reward_sum=0
    
    def reset(self):
        state=random.choice(all_states)
        return state
    
    def get_reward(self,state):
        return state.con
    
    def get_next_state(self, state, action):
        """Computes the newstate.

        Takes a state and an action from the agent and computes its next position.

        Args:
            state: current state
            action: index of an action

        Returns:
            newstate: a tuple (m, n) representing the coordinates of the new position

        """
        #state= get_state(state_location)
        next_state_location  =(state.row+steps[action][0],state.col+steps[action][1])
        next_state=get_state(next_state_location)
        return next_state
    
    
    
    def draw_log(self):
        #data = np.zeros((rows, cols))
        
        if len(self.record_list)>0:            
            for item in self.record_list:                
                #start 
                if item[0].con==2:
                    data[item[0].row, item[0].col]==2
                #destination
                elif item[0].con==1:
                    data[item[0].row, item[0].col]=1
                #cliff
                elif item[0].con==-1:
                    data[item[0].row, item[0].col]=-1
                else:
                    data[item[0].row, item[0].col]=0
                
                annotation_movement[item[0].row,item[0].col]=str(item[1])                
                
                if item[0].row==0 and item[0].col==0:
                    print("update:",item)
            
            
            sns.heatmap(data,cmap=states_colors, annot=annotation_movement, cbar=False, square=True, linewidths=1, fmt='')
            plt.show()
            
    def draw_log(self,data_log):
        #data = np.zeros((rows, cols))
        
        if len(data_log)>0:            
            for item in data_log:                
                #start 
                if item[0].con==2:
                    data[item[0].row, item[0].col]==2
                #destination
                elif item[0].con==1:
                    data[item[0].row, item[0].col]=1
                #cliff
                elif item[0].con==-1:
                    data[item[0].row, item[0].col]=-1
                else:
                    data[item[0].row, item[0].col]=0
                
                annotation_movement[item[0].row,item[0].col]=str(item[1])                
                
                if item[0].row==0 and item[0].col==0:
                    print("update:",item)
            
            
            sns.heatmap(data,cmap=states_colors, annot=annotation_movement, cbar=False, square=True, linewidths=1, fmt='')
            plt.show()
            
    
    def get_valid_actions(self, state):
        """
        

        Parameters
        ----------
        state : State
            DESCRIPTION.

        Returns
            a list of valid actions
        -------
        
        steps={'<':(0,-1),
                   '>':(0,1),
                   '^':(-1,0),
                   'v':(1,0)}

        """
        valid_actions=[]
        if state.row+steps['v'][0]<=3:
            valid_actions.append('v')
        if state.row+steps['^'][0]>=0:
            valid_actions.append('^')
        if state.col+steps['<'][1]>=0:
            valid_actions.append('<')
        if state.col+steps['>'][1]<=11:
            valid_actions.append('>')
        
        #print(state,valid_actions)
        return valid_actions
    
class Agent:
    def __init__(self, env):
        self.env=env
    def __init__(self, env, alpha, epsilon, gamma):
        self.env = env
        self.epsilon = epsilon 
        self.alpha = alpha
        self.gamma = gamma
        self.Q=defaultdict(int)
        self.A=defaultdict(set)
        self.td_list = []
        
    def get_espilon_action(self, state):
        valid_actions= self.env.get_valid_actions(state)
        if np.random.random() > self.epsilon:
            action = self.get_best_action(state)
        else:
            action = self.get_random_action(state)
        return action
    
    def learn(self,state, action, reward, next_state):
        temp_td = self.td(state, action, reward, next_state)
        self.Q[state, action]= self.Q[state,action]+self.alpha*temp_td
        
    def td(self, state, action, reward, next_state):
        max_action  = self.get_best_action(state)
        temp_td= reward + self.gamma*self.Q[state,max_action]-self.Q[state,action]
        self.td_list.append(temp_td)
        return temp_td
        
    def get_best_action(self, state):
        #print(self.Q)
        valid_actions= self.env.get_valid_actions(state)
        max_value=-1
        max_action=valid_actions[0]
        for action in valid_actions:
            next_state=self.env.get_next_state(state, action)
            if max_value<self.Q[next_state, action]:
                max_value=self.Q[next_state, action]
                max_action=action
        #if state ==all_states[0]:
        #    print("state",state,"max action ", max_action, "max Q:", max_value)
        return max_action
    
            
    def get_random_action(self, state):
        valid_actions = self.env.get_valid_actions(state)
        return random.choice(valid_actions)
    
    def log(self,state,action, reward, next_state):
        self.env.record_list.append((state,action, reward, next_state))
        self.env.reward_sum+=reward
    
    
    
def run_random_episode(agent):
    current_state= agent.env.start_state
    while not (current_state.con==-1 or current_state.con==1):
        action = agent.get_random_action(current_state)
        #action = agent.get_espilon_action(current_state)
        next_state=agent.env.get_next_state(current_state,action)
        reward =next_state.con
        #agent.env.log(current_state, action, reward, next_state)
        current_state=next_state 
        agent.log(current_state, action, reward, next_state)
        #print(agent.env.record_list)
    agent.env.draw_log()

def run_epsilon_episode(agent):
    
    for i in range(n_episodes):
        data_log=[]
        current_state= agent.env.start_state
        while  (1):
            if current_state.con==-1:
                print("Falling.end episode ",i)
                break
            if current_state.con==1:
                print("Congrat!")
                break
            
            action = agent.get_espilon_action(current_state)           
            next_state=agent.env.get_next_state(current_state,action)
            
            # print(next_state)
            reward =next_state.con
            #agent.env.log(current_state, action, reward, next_state)
            #update Q
            agent.learn(current_state, action, reward, next_state)
            agent.log(current_state, action, reward, next_state)
            data_log.append((current_state, action, reward, next_state))
            current_state=next_state 
        agent.env.draw_log(data_log)
            
           
            
    #print(agent.env.record_list)        
    #agent.env.draw_log()
    

epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.9
alpha = 0.25
n_episodes = 500
env = CliffWorld(all_states[0])

agent = Agent(env,alpha,epsilon,gamma)
#agent.env.draw_log()
#agent.env.get_valid_actions(all_states[0])
run_epsilon_episode(agent)
plt.plot(agent.td_list)
plt.show

# string_movement=""
# env = CliffWorld(all_states[0])
# string_movement=string_movement+"("+str(env.start_state.x)+","+str(env.start_state.y)+")"
# action ='>'
# string_movement+=action
# next_state=env.get_next_state(all_states[0], '>')
# string_movement=string_movement+"("+str(next_state.x)+","+str(next_state.y)+")"
# print(string_movement)

# print(env.valid_actions(all_states[12]))