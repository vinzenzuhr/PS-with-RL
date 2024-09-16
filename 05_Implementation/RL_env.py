#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dataclasses import dataclass
import os

import torch 
from torch.utils.tensorboard import SummaryWriter

from custom_modules import Training, DataGenerator, GNN


# In[2]:


@dataclass
class TrainingConfig:
    output_dir = "RL_PersSched"
    num_epoch = 1000000 #10'000 needs 1 hour with 3x CPU, 3GB RAM and 1x 1080ti
    max_steps = 28
    batch_size = 1280
    replay_size = 1280
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_every_n_epochs = 2000
    lr = 1e-4
    gamma = 0.99
    hidden_dim = 32
    num_message_layer = 2
    clip_norm = 1

config = TrainingConfig()


# ## Training

# In[3]:


tb_summary = SummaryWriter(config.output_dir, purge_step=0)
os.makedirs(config.output_dir, exist_ok=True)


# In[4]:


dim_employee = 1
dim_shift = 7
gnn = GNN((dim_employee, dim_shift), hidden_dim=config.hidden_dim, num_message_layer=config.num_message_layer, device=config.device)


# In[5]:


optimizer = torch.optim.AdamW(
    gnn.parameters(), 
    lr=config.lr, 
    amsgrad=True
    ) 


# In[6]:


tb_summary.add_scalar("num_parameters", len(gnn.parameters()), 0)
tb_summary.add_scalar("num_epoch", config.num_epoch, 0)
tb_summary.add_scalar("max_steps", config.max_steps, 0)
tb_summary.add_scalar("batch_size", config.batch_size, 0)
tb_summary.add_scalar("replay_size", config.replay_size, 0)
tb_summary.add_scalar("eval_every_n_epochs", config.eval_every_n_epochs, 0)
tb_summary.add_scalar("lr", config.lr, 0)
tb_summary.add_scalar("gamma", config.gamma, 0)
tb_summary.add_scalar("hidden_dim", config.hidden_dim, 0)
tb_summary.add_scalar("num_message_passing", config.num_message_layer, 0)
tb_summary.add_scalar("clip_norm", config.clip_norm, 0)
tb_summary.add_scalar("num_shifts", DataGenerator.get_week_shifts().shape[0], 0)


# In[7]:


training = Training(
    gnn, 
    optimizer, 
    tb_summary, 
    device=config.device,  
    gamma=config.gamma,
    max_steps=config.max_steps, 
    num_epoch=config.num_epoch, 
    batch_size=config.batch_size, 
    replay_size=config.replay_size, 
    eval_every_n_epochs=config.eval_every_n_epochs,
    output_dir=config.output_dir,
    clip_norm=config.clip_norm
) 
training.start_training()


# ## Save as python script

