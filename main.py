
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pygame


#1-set up the environment
#_________________________
env = gym.make('FrozenLake8x8-v1',render_mode='human')  # using FrozenLake environment
STATES = env.observation_space.n     #get number of states
ACTIONS = env.action_space.n         #get number of actions

#2-building the Q-table
#______________________

Q = np.zeros((STATES,ACTIONS))  #create a matrix with all 0 values

EPISODES = 2000     #how many times to run the environment from the beginning
MAX_STEPS = 150     #max number of steps allowed for each run of environment
LEARNING_RATE = 0.81    # a higher rate a faster the model learn
GAMMA = 0.96
RENDER = False


epsilon = 0.9       #start with a 90% chance of picking a random action and 10% to look at Q-table to take the action

rewards = []        #store all rewards
for episode in range(EPISODES):
    state ,_ = env.reset()  #reset the environment to default state
    print('episode: ',episode)
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()    #render the GUI for the environment
        # code to pick action
        if np.random.uniform(0,1)<epsilon:     #picking random value between 0 and 1
            action = env.action_space.sample()  #take random action
        else:
            action = np.argmax(Q[state,:]) #use Q-table to pick max value in column table based on the current value
        new_state, reward, done,_,_ = env.step(action)  #take action , and return for us information about the action
        # updating Q values
        Q[state,action] = Q[state,action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state,action])

        state = new_state

        if done:
            rewards.append(reward)
            epsilon-=0.001
            break   #reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}")
#now you can see the model

#we can plot the training progress and see how the agent improve


def get_average(values):
    return sum(values)/len(values)

avg_reward = []
for i in range(0 , len(rewards),100):
    avg_reward.append((get_average(rewards[i:i+100])))

plt.plot(avg_reward)        #the avg of each 100 episodes
plt.ylabel('average reward')
plt.xlabel('episodes (100\s)')
plt.show()
