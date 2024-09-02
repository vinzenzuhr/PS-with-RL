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
        projection_employees (torch.nn.Linear): Linear layer for projecting employee features.
        projection_shifts (torch.nn.Linear): Linear layer for projecting shift features. 
        num_message_passing (int, optional): Number of iterations to embed nodes with GNN. Defaults to 4.      
    """
    
    def __init__(self, gnn: RGCNConv, projection_employees: torch.nn.Linear, projection_shifts: torch.nn.Linear, num_message_passing: int = 4,) -> None: 
        self.gnn = gnn  
        self.projection_employees = projection_employees
        self.projection_shifts = projection_shifts
        self.num_message_passing = num_message_passing

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
        assignments = state["assigned"]["edge_index"]
        assignments_flipped=torch.vstack((state["assigned"]["edge_index"][1], state["assigned"]["edge_index"][0])) 
        edge_type = torch.zeros(state["assigned"]["edge_index"].shape[1], dtype=torch.int64) 

        emb_employees = self.projection_employees(state.x_dict["employee"]) 
        emb_shifts = self.projection_shifts(state.x_dict["shift"])
        
        for _ in range(self.num_message_passing):
            emb_shifts = self.gnn(
                x=(emb_employees, emb_shifts), 
                edge_index=assignments, 
                edge_type=edge_type
                )

            emb_employees = self.gnn(
                x=(emb_shifts, emb_employees), 
                edge_index=assignments_flipped, 
                edge_type=edge_type
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