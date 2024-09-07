from typing import Tuple

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
        self.max_steps = max_steps
        self.env = env 
        self.env.reset()

    def execute_episode(self) -> list[Tuple[HeteroData, int, float]]:
        """
        Does as many steps until episode is finished or max_steps is reached.
        Returns:
            A list of tuples representing the steps taken in the episode. Each tuple contains:
            - state: The state before taking the action.
            - action: The action taken in the state.
            - reward: The reward received after taking the action.
        """ 
        state, _ = self.env.reset()
        terminated = False
        steps = []
        i = 0
        while not terminated and i < self.max_steps:  
            action, new_state, reward, terminated = self.step() 
            steps.append((state, action, reward)) 
            state = new_state 
            i = i + 1  
        return steps

    def step(self) -> Tuple[int, HeteroData, float, bool]:
        """
        Takes a step in the environment based on the current state.
        Returns:
            Tuple[int, HeteroData, float, bool]: A tuple containing the action taken, the new state, 
                the reward received, and a flag indicating if the episode terminated.
        """
        
        action = self.agent.get_policy(self.env.state).sample().item()  
        state, reward, terminated, _, _ = self.env.step(action) 
        return action, state, reward, terminated