#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # TO DO: Initialize relevant elements
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        p = np.random.rand()
        if p < epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s, :])
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add Dyna update
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r

        if done:
            self.Q_sa[s,a] += self.learning_rate * (r - self.Q_sa[s,a])
        else:
            #next_ = self.Q_sa[s_next, self.select_action(s_next, 0.1)] # Maybe change later
            next_ = np.max(self.Q_sa[s_next]) # MAYBE CHANGE LATER
            self.Q_sa[s, a] += self.learning_rate * (r + self.gamma * next_ - self.Q_sa[s, a])

        for K in range(n_planning_updates):
            # Get all (s, a) pairs with observed transitions
            previous_state_action = np.argwhere(np.sum(self.n, axis=2) > 0)
            if len(previous_state_action) == 0:
                break
            s_prev, a_prev = previous_state_action[np.random.randint(len(previous_state_action))]

            # Compute p_hat(s' | s_prev, a_prev)
            n_total = np.sum(self.n[s_prev, a_prev]) # this is denominator
            if n_total == 0:
                continue
            p_hat = self.n[s_prev, a_prev] / n_total
            s_prime = np.random.choice(self.n_states, p=p_hat)

            # Compute r_hat(s_prev, a_prev, s_prime)
            r_hat = self.R_sum[s_prev, a_prev, s_prime] / self.n[s_prev, a_prev, s_prime]

            # Q-learning update using simulated model
            self.Q_sa[s_prev, a_prev] += self.learning_rate * (r_hat + self.gamma * np.max(self.Q_sa[s_prime]) - self.Q_sa[s_prev, a_prev])

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # TO DO: Initialize relevant elements
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions, n_states))
        self.R_sum = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        p = np.random.rand()
        if p < epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s, :])
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        
        # TO DO: Add Prioritized Sweeping code
        
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a))) 
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue
        self.n[s, a, s_next] += 1
        self.R_sum[s, a, s_next] += r
        if done:
            p = abs(r - self.Q_sa[s, a])
        else:
            p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s,a])
        if p > self.priority_cutoff:
            self.queue.put((-p,(s,a)))

        for K in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (s_p, a_p) = self.queue.get()

            n_total = np.sum(self.n[s_p, a_p])
            if n_total == 0:
                continue
            p_hat = self.n[s_p, a_p] / n_total
            s_prime = np.random.choice(self.n_states, p=p_hat)

            r_hat = self.R_sum[s_p, a_p, s_prime] / self.n[s_p, a_p, s_prime]

            self.Q_sa[s_p, a_p] += self.learning_rate * (r_hat + self.gamma * np.max(self.Q_sa[s_prime]) - self.Q_sa[s_p, a_p])

            for s_bar, a_bar in np.argwhere(self.n[:, :, s_p] > 0):
                r_bar = self.R_sum[s_bar, a_bar, s_p] / self.n[s_bar, a_bar, s_p]
                p = abs(r_bar + self.gamma * np.max(self.Q_sa[s_p]) - self.Q_sa[s_bar, a_bar])
                if p > self.priority_cutoff:
                    self.queue.put((-p, (s_bar, a_bar)))

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
