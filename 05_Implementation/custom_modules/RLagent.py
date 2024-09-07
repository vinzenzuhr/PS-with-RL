from typing import Tuple

import torch
from torch.distributions.categorical import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

class RLagent():
    """
    Agent for choosing an action based on the current state.
    Args:
        gnn (RGCNConv): GNN to encode nodes.   
    """
    
    def __init__(self, gnn: RGCNConv) -> None: 
        self.gnn = gnn   

    def decode(self, emb_employees: torch.tensor, emb_shifts: torch.tensor) -> torch.tensor:
        """
        Uses the embeddings of employees and shifts to calculate the action logits for the policy. 
        Args:
            emb_employees (torch.tensor): Embeddings for employees.
            emb_shifts (torch.tensor): Embeddings for shifts.
        Returns:
            torch.tensor: The logits for the policy.
        """
        num_employees = emb_employees.shape[0]
        num_shifts = emb_shifts.shape[0]

        expanded_shifts = emb_shifts.repeat((num_employees, 1))
        expanded_employees = emb_employees.repeat_interleave(num_shifts, dim=0)
        dot_products = expanded_shifts.mul(expanded_employees).squeeze().sum(dim=1) 
        
        return dot_products

    def encode(self, state: HeteroData) -> Tuple[torch.tensor, torch.tensor]:
        """
        Encodes the given state into embeddings for employees and shifts.
        Args:
            state (HeteroData): The input state containing employee and shift nodes and the assignments.
        Returns:
            Tuple[torch.tensor, torch.tensor]: Embeddings for employees and shifts.
        """

        emb_employees, emb_shifts = self.gnn.forward(
            x=(state.x_dict["employee"], state.x_dict["shift"]),
            edge_index=state["assigned"]["edge_index"]
        )

        return emb_employees, emb_shifts
        
    def get_policy(self, state: HeteroData) -> Categorical:
        """
        Returns the policy for a given state.
        Parameters:
            state (HeteroData): The input state.
        Returns:
            Categorical: The policy distribution containing the logits per action.
        """

        logits = self.decode(*self.encode(state)) 
        policy_distribution = Categorical(logits=logits) 
        return policy_distribution