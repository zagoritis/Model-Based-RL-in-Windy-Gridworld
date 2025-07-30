# Model-Based RL in Windy Gridworld
**Assignment Year: 2024-2025**

This project investigates two model-based reinforcement learning algorithms: **Dyna** and **Prioritized Sweeping**, using a stochastic version of the **Windy Gridworld** environment. The goal is to analyze their performance, sensitivity to planning iterations, and effectiveness compared to a model-free Q-learning baseline.

## Overview
The Windy Gridworld consists of a 10×7 grid with vertical wind effects in specific columns. The agent:
- Starts at position (0, 3).
- Aims to reach position (7, 3) for a reward of +100.
- Receives a reward of -1 at each time step otherwise.
- Faces variable wind strength with a customizable `wind_proportion` parameter.

Implemented RL agents:
- **DynaAgent**: Uses real transitions to learn a model, then simulates updates.
- **PrioritizedSweepingAgent**: Extends Dyna's method by selectively updating impactful states via a priority queue.

## Features
- Dyna and Prioritized Sweeping with ε-greedy exploration.
- Configurable wind stochasticity.
- Q-value and policy visualization.
- Hyperparameter tuning for planning, exploration, and learning rates.
- Evaluation and runtime comparison plots.

## Contributors
- **Emmanouil Zagoritis**
- **Kacper Nizielski**

## References
- Thomas Moerland, part of Introduction to Reinforcement Learning course, Leiden University
