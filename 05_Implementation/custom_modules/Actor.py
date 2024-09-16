from typing import Tuple

import torch
from torch_geometric.data import HeteroData

from custom_modules import RLagent, PersonnelScheduleEnv

class Actor:
    """
    Actor which interacts with the environment. 
    Args:
        agent (Agent): The agent used to choose the action based on the current state.
        env (PersonnelScheduleEnv): The environment in which the actor interacts.
        max_steps (int, optional): The maximum number of steps the actor takes in an episode. Defaults to 20. 
    """ 

    def __init__(self, agent: RLagent, env: PersonnelScheduleEnv, max_steps: int = 20) -> None:
        self.agent = agent
        self.env = env 
        self.max_steps = max_steps

        self.env.reset()

    def execute_episode(self) -> list[Tuple[HeteroData, torch.tensor, int, float]]:
        """
        Does as many steps until episode is finished or max_steps is reached.
        Returns:
            A list of tuples representing the steps taken in the episode. Each tuple contains:
            - state: The state before taking the action.
            - logits: The logits of the policy for the state (used for learning).
            - action: The action taken in the state.
            - reward: The reward received after taking the action.
        """ 
        state, _ = self.env.reset()
        terminated = False
        steps = []
        i = 0 
        while not terminated and i < self.max_steps:
            logits, action, new_state, reward, terminated = self.step()   
            steps.append((state, logits, action, reward)) 
            state = new_state 
            i = i + 1   
        return steps

    def step(self) -> Tuple[int, HeteroData, float, bool]:
        """
        Takes a step in the environment based on the current state.
        Returns:
            Tuple[torch.tensor, int, HeteroData, float, bool]: A tuple containing the logits of the 
                policy for the state, action taken, the new state, the reward received, 
                and a flag indicating if the episode terminated.
        """
        policy = self.agent.get_policy(self.env.state)
        action = policy.sample().item()
        state, reward, terminated, _, _ = self.env.step(action) 
        return policy.logits, action, state, reward, terminated