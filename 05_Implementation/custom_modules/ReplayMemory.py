from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("state", "action", "reward", "future_return"))

class ReplayMemory():
    """
    A class for storing and sampling transitions used for batch training.
    Args:
        capacity (int): The maximum storage capacity. 
    """

    def __init__(self, capacity: int) -> None:
        self.memory=deque([], maxlen=capacity)

    def clear(self) -> None:
        """
        Clears the memory.
        """
        self.memory.clear()

    def push(self, transition: Transition) -> None:
        """
        Add a transition to storage.
        Parameters:
            transition (Transition): The transition to be added. 
        """
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Randomly samples a batch of transitions from storage.
        Args:
            batch_size (int): The number of transitions to sample.
        Returns:
            list[Transition]: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """
        Returns the number of stored objects.
        Returns:
            int: Number of stored objects.
        """
        return len(self.memory)