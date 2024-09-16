from collections import namedtuple 
import sys

from tqdm.auto import tqdm
import torch
from torch import distributions
from torch.cuda import device
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import RGCNConv

from custom_modules import ReplayMemory, ActorMemoryWrapper, DataGenerator, PersonnelScheduleEnv, RLagent, Actor

Transition = namedtuple("Transition", ("state", "logits", "action", "reward", "future_return"))

class Training():
    class Training:
        """
        Class for training the encoder models of the reinforcement learning agent. 
        Args:
            gnn (RGCNConv): GNN to encode nodes. 
            optimizer (Optimizer): Optimizer for GNN and projection layers. 
            tensorboard (SummaryWriter): Tensorboard writer for logging.
            device (device): Device to run the training on.    
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 20.
            num_epoch (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 128.
            replay_size (int, optional): Storage size of replay memory. Defaults to 10000.
            eval_every_n_epochs (int, optional): Evaluate the agent every n epochs. Defaults to 20.
            output_dir (str, optional): Output directory for saving model weights. Defaults to ".". 
        """ 
    
    def __init__(
            self, 
            gnn: RGCNConv,  
            optimizer: Optimizer, 
            tensorboard: SummaryWriter, 
            device: device,  
            gamma: float = 0.9,
            max_steps: int = 20, 
            num_epoch: int = 100, 
            batch_size: int = 128, 
            replay_size: int = 10000, 
            eval_every_n_epochs: int = 20, 
            output_dir: str = ".",
            clip_norm: float = 1.0
            ):
        self.gnn = gnn  
        self.optimizer = optimizer 
        self.tensorboard = tensorboard
        self.device = device  
        self.max_steps = max_steps
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.memory = ReplayMemory(replay_size)
        env = PersonnelScheduleEnv(
            employees=DataGenerator.get_random_employees(), 
            shifts=DataGenerator.get_week_shifts(), 
            assignments=DataGenerator.get_empty_assignments(), 
            device=device
            )
        self.agent = RLagent(gnn, env) 
        self.actor = ActorMemoryWrapper(Actor(self.agent, env=env, max_steps=max_steps), self.memory, gamma=gamma)
        self.eval_every_n_epochs = eval_every_n_epochs
        self.output_dir = output_dir
        self.clip_norm = clip_norm
        self.best_eval_future_returns_employee_5 = -sys.maxsize
    
    def _gradient_update(self) -> None:
        """
        Performs a gradient update step of the GNN encoder models. 
        """
        if len(self.memory) < self.batch_size:
            return 

        transitions = self.memory.sample(self.batch_size)

        #converts batch_array of Transitions to Transition of batch_arrays
        batch = Transition(*zip(*transitions))
        
        logits_batch = torch.nn.utils.rnn.pad_sequence(
            batch.logits,  
            batch_first=True,  
            padding_value=-float("inf")  
            ).to(self.device)  
        action_batch = torch.tensor(batch.action, dtype=torch.int).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
        future_returns_batch = torch.tensor(batch.future_return, dtype=torch.float).to(self.device)

        policy_distribution = distributions.categorical.Categorical(logits=logits_batch) 
        log_probs = policy_distribution.log_prob(action_batch)
        
        objective = (-log_probs * future_returns_batch).sum() / self.batch_size 

        self.optimizer.zero_grad() 
        objective.backward()
        if self.clip_norm > 0:
            gnn_gradient_norm = torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.clip_norm)
        self.optimizer.step()

        self.progress_bar.set_postfix({
            f"obj": objective.detach().item(), 
            f"rew": reward_batch.mean().detach().item(),
            f"fut_ret": future_returns_batch.mean().detach().item(),
            f"grad": gnn_gradient_norm.detach().item()
            })

        self.memory.clear()

        self.tensorboard.add_scalar("train_objective", objective.detach().cpu(), self.epoch)
        self.tensorboard.add_scalar("train_avg_reward", reward_batch.mean().detach().cpu(), self.epoch)
        self.tensorboard.add_scalar("train_avg_future_returns", future_returns_batch.mean().detach().cpu(), self.epoch)
        self.tensorboard.add_scalar("gnn_gradient_norm", gnn_gradient_norm.detach().cpu(), self.epoch)

    def evaluation(self) -> None:
        """
        Performs evaluation of the model's performance on different number of employees and saves best model. 
        """  
        num_employees = 5
        env = PersonnelScheduleEnv(
            employees=DataGenerator.get_random_employees(num_employees, num_employees), 
            shifts=DataGenerator.get_week_shifts(), 
            assignments=DataGenerator.get_empty_assignments(), 
            device=self.device
            )
        evaluation_steps = ReplayMemory(self.max_steps)
        evaluation_agent = RLagent(self.gnn, env)
        evaluation_actor = ActorMemoryWrapper(
            Actor(evaluation_agent, env=env, max_steps=self.max_steps), 
            evaluation_steps
            )
        evaluation_actor.sample_episode() 
        eval_future_returns_employee_5 = evaluation_steps.memory[0].future_return
        self.tensorboard.add_scalar("future_returns_employee_5", eval_future_returns_employee_5, self.epoch)    
        self.tensorboard.add_scalar("num_assigned_shifts_employee_5", env.get_num_staffed_shifts(), self.epoch)
        
        if eval_future_returns_employee_5 > self.best_eval_future_returns_employee_5:
            self.gnn.save(self.output_dir + "/gnn_weights") 
        
    def start_training(self) -> None:
        """
        Starts the training process.  
        """
        self.progress_bar = tqdm(range(self.num_epoch))
        for epoch in range(self.num_epoch):
            self.progress_bar.set_description(f"Epoch {epoch}")
            self.epoch = epoch
            env = PersonnelScheduleEnv(
                employees=DataGenerator.get_random_employees(2,10), 
                shifts=DataGenerator.get_week_shifts(), 
                assignments=DataGenerator.get_empty_assignments(), 
                device=self.device
                ) 
            self.actor.env = env
            self.agent.env = env 
            self.actor.sample_episode() 
            self._gradient_update() 

            if epoch % self.eval_every_n_epochs == 0:
                self.evaluation()

            self.progress_bar.update(1)