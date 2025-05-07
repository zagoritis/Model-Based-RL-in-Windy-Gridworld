#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updates = [1,3,5] 
    
    ''' Dyna '''
    mean_returns = run_repetition("dyna", n_repetitions, n_timesteps, eval_interval, n_planning_updates, learning_rate, epsilon, gamma)
    plot = LearningCurvePlot(title='Dyna')
    plot.add_curve(np.arange(0, n_timesteps, eval_interval), mean_returns, label='Dyna')
    plot.set_ylim(-10, 0)
    plot.add_hline(-1, 'Optimal')
    plot.save('dyna.png')
    
    ''' Prioritized Sweeping '''
    ###

def run_repetition(agent_type, n_repetitions, n_timesteps, eval_interval, n_planning_updates, learning_rate=0.1, epsilon=0.1, gamma=1.0):
    total_returns = []
    env = WindyGridworld()
    for _ in range(n_repetitions):
        s = env.reset()
        if agent_type == 'dyna':
            agent = DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma)
        elif agent_type == 'prioritizedsweeping':
            agent = PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma)

        for _ in range(n_timesteps):
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

            if eval_interval == 250:
                 total_returns.append(agent.evaluate(env))
    return np.mean(total_returns, axis=0)

if __name__ == '__main__':
    experiment()
