import os

import tqdm.auto as tqdm
import numpy as np

import torch
import wandb

from molspace.predictors.drug_likeness import predict_log_p


def step(state, graph, action):
    next_state = graph.data[action]
    prev_molecule = graph.data.get_rdkit(state)
    next_molecule = graph.data.get_rdkit(next_state)
    reward = predict_log_p(next_molecule) - predict_log_p(prev_molecule)
    return next_state, reward, False, {}


def train_step(agent, graph, use_wandb=False, train_model=True):

    os.makedirs("./test/test_results", exist_ok=True)
    state = np.random.randint(len(graph))
    progress_bar = tqdm.trange(1000)

    total_reward = 0

    for time in progress_bar:
        action = agent.act(state)

        next_state, reward, done, debugging_output = step(state, graph, action)
        total_reward += reward
        state = next_state

        if train_model and (time + 1) % 1000 == 0:
            loss_v, loss_p = agent.replay()
            if use_wandb:
                wandb.log({'Value Loss': loss_v, 'Policy Loss': loss_p})
            torch.save(agent.model.state_dict(), f"best-model.h5")

        progress_bar.set_postfix(total_reward=total_reward, time=time)
        if done:
            progress_bar.set_postfix(total_reward=total_reward, time=time)
            progress_bar.close()

    if train_model:
        loss_v, loss_p = agent.replay()
        if use_wandb:
            wandb.log({'Value Loss': loss_v, 'Policy Loss': loss_p})
        torch.save(agent.model.state_dict(), f"best-model.h5")

    return True
