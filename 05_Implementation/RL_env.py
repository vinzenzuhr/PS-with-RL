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
    lr=1e-4

config = TrainingConfig()


# ## Training

# In[4]:


tb_summary = SummaryWriter(config.output_dir, purge_step=0)
os.makedirs(config.output_dir, exist_ok=True)


# In[5]:


gnn_employees = RGCNConv(in_channels = (1,7), out_channels=2, num_relations=1).to(config.device)
gnn_shifts = RGCNConv(in_channels = (7,1), out_channels=2, num_relations=1).to(config.device)


# In[6]:


optimizer_employees = torch.optim.AdamW(gnn_employees.parameters(), lr=config.lr, amsgrad=True)
optimizer_shifts = torch.optim.AdamW(gnn_shifts.parameters(), lr=config.lr, amsgrad=True)


# In[7]:


training = Training(
    gnn_employees, 
    gnn_shifts, 
    optimizer_employees,
    optimizer_shifts,
    tb_summary, 
    device=config.device, 
    max_steps=config.max_steps, 
    num_epoch=config.num_epoch, 
    batch_size=config.batch_size, 
    replay_size=config.replay_size, 
    eval_every_n_epochs=config.eval_every_n_epochs,
    output_dir=config.output_dir
)
training.start_training()


# ## Save as python script

