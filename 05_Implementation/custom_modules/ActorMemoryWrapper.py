from collections import namedtuple 

from custom_modules import Actor, ReplayMemory

Transition = namedtuple("Transition", ("state", "action", "reward", "future_return"))

class ActorMemoryWrapper():
    """
    A wrapper class to the Actor Class, which calculates future_returns and saves all steps in a replay memory.
    Args:
        actor (Actor): The actor object.
        memory (ReplayMemory): The replay memory object. 
        gamma (float, optional): Discount factor for future rewards. Defaults to 0.9.
    """
    
    def __init__(self, actor: Actor, memory: ReplayMemory, gamma: float = 0.9):
        self.actor = actor
        self.memory = memory
        self.gamma = gamma

    def __getattr__(self, name: str):
        """
        Overrides the default behavior when an attribute is not found.
        Args:
            name (str): The name of the attribute being accessed.
        Returns:
            Any: The value of the attribute if found in the actor object.
        """ 
        return getattr(self.actor, name)

    def sample_episode(self): 
        """
        Executes an episode using the actor, calculates future_returns and stores the steps in the memory. 
        """  
        steps = self.actor.execute_episode() 
        transitions = []  
        future_returns = 0
        for step in reversed(steps):
            future_returns = self.gamma*future_returns + step[2] 
            transitions.append(Transition(*step, future_returns)) 
        for transition in reversed(transitions): 
            self.memory.push(transition)
            
            
            