import random

import torch

class DataGenerator():
    """
    A class that generates data for employee, shifts and assignments. 
    """
    
    def get_random_employees(min_n: int = 1, max_n: int = 5) -> torch.Tensor:
        """
        Generates a random number of employees and returns a tensor representing their features (no features yet). 
        Args:
            min_n (int, optional): The minimum number of employees to generate. Defaults to 1.
            max_n (int, optional): The maximum number of employees to generate. Defaults to 5.
        Returns:
            torch.Tensor: A tensor representing the features of the generated employees.
        """
        
        n = random.randint(min_n, max_n)
        return torch.tensor([[1]], dtype=torch.float).expand((n,1)) 
    
    def get_week_shifts() -> torch.Tensor:
        """
        Returns a tensor representing the two shifts for each day of the week.
        Returns:
            torch.Tensor: A tensor of shape (14, 7) where each row represents a shift for a specific day of the week.
                          The columns represent the following shifts:
                          - Column 0: Monday, day shift
                          - Column 1: Monday, night shift
                          - Column 2: Tuesday, day shift
                          - Column 3: Tuesday, night shift
                          - Column 4: Wednesday, day shift
                          - Column 5: Wednesday, night shift
                          - Column 6: Thursday, day shift
                          - Column 7: Thursday, night shift
                          - Column 8: Friday, day shift
                          - Column 9: Friday, night shift
                          - Column 10: Saturday, day shift
                          - Column 11: Saturday, night shift
                          - Column 12: Sunday, day shift
                          - Column 13: Sunday, night shift
        """

        shifts = list()
        shifts.append(torch.tensor([0,0,0,0,0,0,0])) # Monday, day
        shifts.append(torch.tensor([0,0,0,0,0,0,1])) # Monday, night
        shifts.append(torch.tensor([1,0,0,0,0,0,0])) # Tuesday, day
        shifts.append(torch.tensor([1,0,0,0,0,0,1])) # Tuesday, night
        #shifts.append(torch.tensor([0,1,0,0,0,0,0])) # Wednesday, day
        #shifts.append(torch.tensor([0,1,0,0,0,0,1])) # Wednesday, night
        #shifts.append(torch.tensor([0,0,1,0,0,0,0])) # Thursday, day
        #shifts.append(torch.tensor([0,0,1,0,0,0,1])) # Thursday, night
        #shifts.append(torch.tensor([0,0,0,1,0,0,0])) # Friday, day
        #shifts.append(torch.tensor([0,0,0,1,0,0,1])) # Friday, night
        #shifts.append(torch.tensor([0,0,0,0,1,0,0])) # Saturday, day
        #shifts.append(torch.tensor([0,0,0,0,1,0,1])) # Saturday, night
        #shifts.append(torch.tensor([0,0,0,0,0,1,0])) # Sunday, day
        #shifts.append(torch.tensor([0,0,0,0,0,1,1])) # Sunday, night
        shifts = torch.vstack(shifts)
        shifts = shifts.to(torch.float)
        return shifts
    
    def get_empty_assignments() -> torch.Tensor:
        """
        Returns an empty tensor representing assignments.
        Returns:
            torch.Tensor: An empty tensor with shape (2, 0).
        """
        
        return torch.tensor([[],[]], dtype=torch.int64)