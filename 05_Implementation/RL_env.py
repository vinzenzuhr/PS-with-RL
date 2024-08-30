#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass
import os

import torch 
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import RGCNConv

from custom_modules import Training


# In[2]:


@dataclass
class TrainingConfig:
    output_dir = "RL_PersSched"
    num_epoch = 220000 #10'000 needs 1 hour with 3x CPU, 3GB RAM and 1x 1080ti
    max_steps = 20
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_size = 10000
    eval_every_n_epochs = 20
    lr = 1e-4
    gamma = 0.9
    hidden_dim = 3
    num_message_passing = 4

config = TrainingConfig()


# ## Training

# In[3]:


tb_summary = SummaryWriter(config.output_dir, purge_step=0)
os.makedirs(config.output_dir, exist_ok=True)


# In[4]:


gnn = RGCNConv(in_channels = (config.hidden_dim, config.hidden_dim), out_channels=config.hidden_dim, num_relations=1).to(config.device)
dim_employee = 1
projection_employees = torch.nn.Linear(dim_employee, config.hidden_dim).to(config.device)
dim_shift = 7
projection_shifts = torch.nn.Linear(dim_shift, config.hidden_dim).to(config.device)


# In[6]:


optimizer = torch.optim.AdamW(
    list(gnn.parameters()) + list(projection_employees.parameters()) + list(projection_shifts.parameters()), 
    lr=config.lr, 
    amsgrad=True
    ) 


# In[7]:


training = Training(
    gnn, 
    optimizer, 
    projection_employees,
    projection_shifts,
    tb_summary, 
    device=config.device,
    num_message_passing=config.num_message_passing, 
    gamma=config.gamma,
    max_steps=config.max_steps, 
    num_epoch=config.num_epoch, 
    batch_size=config.batch_size, 
    replay_size=config.replay_size, 
    eval_every_n_epochs=config.eval_every_n_epochs,
    output_dir=config.output_dir
)
training.start_training()


# ## Save as python script

