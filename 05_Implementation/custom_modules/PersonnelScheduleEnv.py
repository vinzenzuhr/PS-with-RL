from collections import Counter
import copy
import itertools
from typing import Tuple

import gymnasium as gym
import torch
from torch_geometric.data import HeteroData

class PersonnelScheduleEnv(gym.Env):
    """
    RL environment for personnel scheduling.
    Args:
        employees (torch.tensor): Features representing the employees.
        shifts (torch.tensor): Features representing the shifts.
        assignments (torch.tensor): Tensor representing the shift assignments.
        device (torch.device): Device to be used for computations. 
        num_employee_per_shift (int, optional): Number of employees which are needed per shift. Defaults to 2.
    """ 
    
    def __init__(
            self, 
            employees: torch.tensor, 
            shifts: torch.tensor, 
            assignments: torch.tensor, 
            device: torch.device,
            num_employee_per_shift: int = 2
            ) -> None:    
        self.device = device
        self.num_employee_per_shift = num_employee_per_shift
        self.initial_state=HeteroData()
        self.initial_state["employee"].x = employees
        self.initial_state["shift"].x = shifts
        self.initial_state["employee", "assigned", "shift"].edge_index = assignments
        self.initial_state = self.initial_state.to(self.device)
        assert self.initial_state.validate(raise_on_error=True), "Data validation not successful"

        self.edge_space = list(itertools.product(range(employees.shape[0]),range(shifts.shape[0])))
        self.action_space = gym.spaces.Discrete(len(self.edge_space))
        self.num_employees = self.initial_state["employee"].x.shape[0]
        self.num_shifts = self.initial_state["shift"].x.shape[0]

    def _get_num_consecutive_violations(self) -> int:
        """
        Calculates the number of consecutive violations in the current planning.
        Returns:
            int: The number of consecutive violations.
        """

        planning = self.get_current_planning()
        num_consecutive_violations = 0
        for i in range(self.num_shifts-1): 
            num_consecutive_violations = num_consecutive_violations + (
                Counter(planning[i]) - (Counter(planning[i]) - Counter(planning[i+1]))
                ).total()  
        return num_consecutive_violations

    def are_all_shifts_staffed(self) -> bool:
        """
        Checks if all shifts have enough personnel assigned.
        Returns:
            bool: True if all shifts have enough personnel assigned, False otherwise.
        """
        staffed = True
        planning = self.get_current_planning()
        for i in range(self.num_shifts): 
            if len(planning[i]) < self.num_employee_per_shift:
                staffed = False
                break
        return staffed
    
    def idx2edge(self, idx: int) -> Tuple[int,int]:
        """
        Converts an action index to an edge in the graph.
        Args:
            idx (int): The action index of the edge.
        Returns:
            Tuple[int,int]: The edge represented by the action index.
        """

        return self.edge_space[idx]

    def edge2idx(self, edge: Tuple[int,int]) -> int:
        """
        Converts an edge to its corresponding action index.
        Args:
        - edge (Tuple[int,int]): The edge to lookup the action index for.
        Returns:
        - int: The action index of the edge.
        """  
        return self.edge_space.index(edge)

    def get_num_staffed_shifts(self) -> int:
        """
        Returns the number of shifts which have enough personnel assigned.
        Returns:
            int: The number staffed shifts.
        """ 
        planning = self.get_current_planning()
        num_staffed_shifts = 0
        for i in range(self.num_shifts): 
            if len(planning[i]) >= self.num_employee_per_shift:
                num_staffed_shifts = num_staffed_shifts + 1
        return num_staffed_shifts
            
    def get_current_planning(self) -> dict:
        """
        Returns the current planning as a dictionary for visualization.
        Returns:
            dict: A dictionary representing the current planning. 
                The keys are the shifts and the values are lists of assigned personnel.
        """

        planning = dict() 
        for i in range(self.num_shifts):
            planning[i] = list()
        for i in torch.arange(self.state["assigned"].edge_index.shape[1]): 
            edge = self.state["assigned"].edge_index[:, i]
            planning[edge[1].item()].append(edge[0].item())
        return planning

    def info(self) -> dict:
        """
        Placeholder for additional environment information.
        Returns:
            A dictionary containing additional environment information. 
        """

        return {}
        
    def step(self, action: int) -> tuple[HeteroData, float, bool, bool, dict]:
        """
        Assigns a shift to an employee based on the given action. 
        Args:
            action (int): The index of the assignment to be made.
        Returns:
            tuple[HeteroData, float, bool, bool, dict]: A tuple containing the updated state, the reward, 
                a boolean indicating if the episode is terminated, a boolean indicating if the episode is truncated, 
                and additional information.
        """

        if not self.action_space.contains(action): 
            return (self.state, self.reward(), self.terminated(), self.truncated(), self.info())

        action_edge = torch.tensor(self.idx2edge(action)).to(self.device)
        mask = torch.ones(self.state["assigned"].edge_index.shape[1], dtype=torch.bool)
        # check if shift exists 
        for i in torch.arange(self.state["assigned"].edge_index.shape[1]): 
            if torch.equal(self.state["assigned"].edge_index[:,i], action_edge):
                mask[i] = False
        # remove # if removing is allowed
        #self.state["assigned"].edge_index = self.state["assigned"].edge_index[:,mask] 

        #print(self.state["assigned"].edge_index.requires_grad)
        
        # if no shift exists, create one
        if not (~mask).any():
            self.state["assigned"].edge_index = torch.hstack((self.state["assigned"].edge_index, action_edge[:,None])) 
            
        return (copy.deepcopy(self.state), self.reward(), self.terminated(), self.truncated(), self.info())

    def reset(self, seed: int = None) -> tuple[HeteroData, dict]:
        """
        Resets the environment to its initial state.
        Parameters:
            seed (int): The random seed for reproducibility. Defaults to None.
        Returns:
            tuple[HeteroData, dict]: A tuple containing the initial state and additional environment information.
        """

        super().reset(seed=seed)
        self.state = copy.deepcopy(self.initial_state)
        return (copy.deepcopy(self.state), self.info())

    def reward(self) -> float:
        """
        Calculates the reward for the current state of the environment.
        Returns:
            float: The calculated reward value.
        """

        reward = 0
        factor = (1/(self.num_shifts-1))
        #reward = reward + factor*self.get_num_staffed_shifts()
        reward = reward - factor*self._get_num_consecutive_violations()
        if self.are_all_shifts_staffed():
            reward = reward + 1
        return reward

    def terminated(self) -> bool:
        """
        Check if every shift has enough personnel assigned or no actions are left.
        Returns:
            bool: True if all shifts are staffed or no actions are left, False otherwise.
        """
        terminated = self.are_all_shifts_staffed()
        if len(self.edge_space) == self.state["assigned"].edge_index.shape[1]:
            terminated = True

        return terminated

    def truncated(self) -> bool:
        """
        Placeholder for backwards compatibility. 
        Returns:
            bool: False.
        """

        return False