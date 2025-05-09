#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""

import time
import numpy as np
import pandas as pd
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_repetitions = 20
    n_timesteps = 10001
    eval_interval = 250
    learning_rate = 0.2
    epsilon = 0.1
    gamma = 1.0
    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]

    def create_plot(algorithm, algorithm_prefix):
        plot = LearningCurvePlot(title=f'{algorithm_prefix} Algorithm')
        for n_planning_updates in n_planning_updatess:
            mean_returns, _ = run_repetitions(algorithm, n_repetitions, n_timesteps, eval_interval, wind_proportion, n_planning_updates, learning_rate, epsilon, gamma)
            smooth_plot = smooth(mean_returns, window=5)
            plot.add_curve(np.arange(0, n_timesteps, eval_interval), smooth_plot, label=f'{algorithm}{n_planning_updates} plans')
        mean_returns_qlearning, _ = run_repetitions("dyna", n_repetitions, n_timesteps, eval_interval, wind_proportion, 0, learning_rate, epsilon, gamma)
        baseline_value = mean_returns_qlearning[-1]
        plot.add_hline(baseline_value, label='Q-learning baseline')
        plot.set_ylim(-100, 100)
        plot.save(f'./figures/{algorithm_prefix}_{wind_proportion}.png')

    for wind_proportion in wind_proportions:
        create_plot("dyna", 'Dyna')
        create_plot("prsw", 'Prioritized Sweeping')

    def create_comparison_plot(environment, wind_proportion, n_planning_dyna, n_planning_prsw):
        plot = LearningCurvePlot(title=f'{environment} Comparison')
        mean_returns_dyna, mean_time_dyna = run_repetitions('dyna', n_repetitions, n_timesteps, eval_interval, wind_proportion, n_planning_dyna, learning_rate, epsilon, gamma)
        mean_returns_prsw, mean_time_prsw = run_repetitions('prsw', n_repetitions, n_timesteps, eval_interval, wind_proportion, n_planning_prsw, learning_rate, epsilon, gamma)
        smooth_plot_dyna = smooth(mean_returns_dyna, window=5)
        smooth_plot_prsw = smooth(mean_returns_prsw, window=5)
        plot.add_curve(np.arange(0, n_timesteps, eval_interval), smooth_plot_dyna, label='dyna5 plans')
        plot.add_curve(np.arange(0, n_timesteps, eval_interval), smooth_plot_prsw, label='prsw5 plans')
        mean_returns_qlearning, mean_time_qlearning = run_repetitions("dyna", n_repetitions, n_timesteps, eval_interval, wind_proportion, 0, learning_rate, epsilon, gamma)
        baseline_value_dyna = mean_returns_qlearning[-1]
        plot.add_hline(baseline_value_dyna, label='Q-learning baseline')
        plot.set_ylim(-100, 100)
        plot.save(f'./figures/{environment}_Comparison.png')
        return mean_time_dyna, mean_time_prsw, mean_time_qlearning

    mean_time_dyna1, mean_time_prsw1, mean_time_qlearning1 = create_comparison_plot('Deterministic', 1.0, 5, 5)
    mean_time_dyna2, mean_time_prsw2, mean_time_qlearning2 = create_comparison_plot('Stochastic', 0.9, 3, 5)
    df = pd.DataFrame(
        [[mean_time_dyna1, mean_time_prsw1, mean_time_qlearning1], [mean_time_dyna2, mean_time_prsw2, mean_time_qlearning2]],
        index=['Stochastic', 'Deterministic'],
        columns=['Dyna', 'Prioritized Sweeping', 'Q-Learning']
    )
    print(df)

def run_repetitions(agent_type, n_repetitions, n_timesteps, eval_interval, wind_proportion, n_planning_updates, learning_rate=0.1, epsilon=0.1, gamma=1.0):
    total_returns, total_times = [], []
    env = WindyGridworld(wind_proportion=wind_proportion)
    for _ in range(n_repetitions):
        start_time = time.time()
        if agent_type == 'dyna':
            agent = DynaAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma)
        elif agent_type == 'prsw':
            agent = PrioritizedSweepingAgent(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma)
        s = env.reset()
        eval_returns = []
        for i in range(n_timesteps):
            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)
            if done:
                s = env.reset()
            else:
                s = s_next
            if i % eval_interval == 0:
                eval_returns.append(agent.evaluate(env))
        total_returns.append(eval_returns)
        total_times.append(time.time() - start_time)
    return np.mean(total_returns, axis=0), np.mean(total_times, axis=0)

if __name__ == '__main__':
    experiment()