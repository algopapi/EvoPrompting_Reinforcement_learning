# EvoPrompt for Reinforcement Learning Architectures

Using LLM's to improve their underlying architecture through an Evolutionary Algorithm is, besides being poetically satisfying, also proven to be quite effective. 

EvoPrompting on Reinforcement Learning Architectures aims to create novel reinforcement learning (RL) algorithms by incorporating an evolutionary algorithm with a large language model (LLM) as a crossover operator.

This project is based on the paper: [EvoPrompting: Language Models for Code-Level Neural Architecture Search](https://arxiv.org/pdf/2302.14838.pdf) by Angelica Chen, David M. Dohan, and David R. So.

The authors successfully demonstrate the application of an evolutionary algorithm with an LLM to create novel machine learning architectures for the MNIST problem. In this project, we attempt to replicate their success for reinforcement learning algorithms.

## Overview

- Replace task T and dataset D with task T and environment E (CartPole-v1).
- Use minimal implementations of popular algorithms as seed code (AC2, ACER, DQN, PPO, REINFORCE).
- Adapt the LLM tuning approach due to limited access to large parameter embeddings.

## Project Structure

### Seed Implementations

We begin with minimal implementations of fan-favorite reinforcement learning algorithms:

1. **AC2** - Advantage Actor-Critic
2. **ACER** - Actor-Critic with Experience Replay
3. **DQN** - Deep Q-Network
4. **PPO** - Proximal Policy Optimization
5. **REINFORCE** - REINFORCE Algorithm

These seed implementations will serve as the starting point for the evolutionary algorithm.

### Environment

The chosen environment for this project is `CartPole-v1` from the OpenAI Gym.

### LLM Tuning

Due to the lack of access to the embeddings of a 65-billion parameter LLM, we are currently working on an alternative method for fine-tuning GPT-3.

## Results

Results and insights gained from the evolutionary algorithm will be uploaded as soon as they are available.

## Acknowledgements

Special thanks to the authors of the original paper, Angelica Chen, David M. Dohan, and David R. So, for their work and inspiration.
