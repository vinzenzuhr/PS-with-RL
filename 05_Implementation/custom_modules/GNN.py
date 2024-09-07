from typing import Tuple

import torch
from torch_geometric.nn import RGCNConv

class GNN:
    """
    Graph Neural Network (GNN) class.
    Args:
        input_dim (Tuple[int, int]): The input dimensions of the source and target nodes.
        hidden_dim (int): The dimension of the hidden layers.
        num_message_layer (int): The number of message passing layers.
        device: The device to run the GNN on.
        bidirectional_weights (bool, optional): If true the weights are shared for both edge direction. Defaults to False. 
    """ 
    def __init__(
            self, 
            input_dim: Tuple[int, int], 
            hidden_dim: int, 
            num_message_layer: int, 
            device: torch.cuda.device, 
            bidirectional_weights: bool = False
            ) -> None:
        self.device = device
        self.bidirectional_weights = bidirectional_weights
        self.projection1 = torch.nn.Linear(input_dim[0], hidden_dim).to(device)
        self.projection2 = torch.nn.Linear(input_dim[1], hidden_dim).to(device)  
        self.conv_layers = list()
        for _ in range(num_message_layer):
            self.conv_layers.append(
                RGCNConv(in_channels = hidden_dim, out_channels=hidden_dim, 
                         num_relations=1 if bidirectional_weights else 2).to(device)
            )

    def load(self, path: str) -> None:
        """
        Loads the model state from the specified path.
        Args:
            path (str): The path to the saved model state. 
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.projection1.load_state_dict(state_dict["projection1"])
        self.projection2.load_state_dict(state_dict["projection2"])
        for i, conv in enumerate(self.conv_layers):
            conv.load_state_dict(state_dict[f"conv_{i}"])

    def forward(self, x: Tuple[torch.tensor, torch.tensor], edge_index: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Performs the forward pass of the GNN module.
        Args:
            x (Tuple[torch.tensor, torch.tensor]): Input tensors for source and target node features.
            edge_index (torch.tensor): Tensor representing the edges.
        Returns:
            Tuple[torch.tensor, torch.tensor]: Embeddings of the input nodes.
        """
        # Flipping the edge index for the opposite direction
        edge_index_flipped = torch.vstack((edge_index[1], edge_index[0]))
        # edge type is 0 for the forward direction and 1 for the backward direction
        edge_type = torch.zeros(edge_index.shape[1], dtype=torch.int64)
        edge_type_opposite = edge_type.clone()
        if not self.bidirectional_weights:
            edge_type_opposite = edge_type_opposite + 1 

        emb_1 = self.projection1(x[0])
        emb_2 = self.projection2(x[1]) 
        for conv in self.conv_layers:
            emb_1_new = conv(x=(emb_2, emb_1), edge_index=edge_index_flipped, edge_type=edge_type)
            emb_2_new = conv(x=(emb_1, emb_2), edge_index=edge_index, edge_type=edge_type_opposite)
            emb_1 = emb_1_new
            emb_2 = emb_2_new 

        return emb_1, emb_2

    def parameters(self) -> list[torch.nn.parameter.Parameter]:
        """
        Returns a list of all the parameters in the GNN module.
        Returns:
            list[torch.nn.parameter.Parameter]: A list of all the parameters.
        """

        params = list(self.projection1.parameters()) + list(self.projection2.parameters())  
        for conv in self.conv_layers:
            params = params + list(conv.parameters()) 
        return params
    
    def save(self, path: str) -> None:
        """
        Saves the state of the GNN model to a file.
        Args:
            path (str): The path to save the model state. 
        """
        state_dict = {
            "projection1": self.projection1.state_dict(),
            "projection2": self.projection2.state_dict()
        }
        for i, conv in enumerate(self.conv_layers):
            state_dict[f"conv_{i}"] = conv.state_dict()
        torch.save(state_dict, path)

        