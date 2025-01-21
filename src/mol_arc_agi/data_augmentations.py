import numpy as np
from typing import List, Dict, Protocol

# Use the strategy pattern to define the augmentation strategies so that they can be easily extended
# All data augmentation strategies should implement this protocol i.e have augment and reverse methods
class AugmentationStrategy(Protocol):
    def augment(self, data: np.array) -> np.array:
        ...

    def reverse(self, data: np.array) -> np.array:
        ...

# Augmentation implementations
# TODO: implement
class TransposeAugmentation:
    def augment(self, data: np.array) -> np.array:
        ...

    def reverse(self, data: np.array) -> np.array:
        ...

class Rotate90DegreesAugmentation:
    def augment(self, data: np.array) -> np.array:
        ...

    def reverse(self, data: np.array) -> np.array:
        ...

class PermuteColorsAugmentation:
    def __init__(self, include_background: bool = True, background_color: int = 0):
        """
        Initialize the PermuteColorsAugmentation with the option to include the background color.
        
        :param include_background: Whether to include the background color in the permutation.
        :param background_color: The background color value (if known).
        """
        self.include_background = include_background
        self.background_color = background_color
        self.permutation = None #TODO: implement a permutation generator

    def augment(self, data: np.array) -> np.array:
        ...

    def reverse(self, data: np.array) -> np.array:
        ...   

class DataAugmentationManager:  
    def __init__(self):
        # Dict format task_id: ordered list of augmentation strategies
        self.augmentations: Dict[str, List[AugmentationStrategy]] = {} 
        #TODO: implement tracking multiple sequences of augmentations being applied to the original task e.g 1. transpose, rotate 2. rotate, rotate, permute ...

    def apply_augmentation(self, task_id: str, data: np.array, strategy: AugmentationStrategy) -> str:
        """
        Apply a given augmentation strategy to the data for a specific task.

        Args:
            task_id (str): The ID of the task.
            data (str): The data to be augmented.
            strategy (AugmentationStrategy): The augmentation strategy to apply.

        Returns:
            str: The augmented data.
        """
        self._ensure_task_exists(task_id)
        augmented_data = strategy.augment(data)
        self._add_augmentation(task_id, strategy)
        return augmented_data

    def _ensure_task_exists(self, task_id: str) -> None:
        if task_id not in self.augmentations:
            self.augmentations[task_id] = []

    def _add_augmentation(self, task_id: str, strategy: AugmentationStrategy) -> None:
        self._ensure_task_exists(task_id)
        """
        Add an augmentation strategy to a specific task.

        Args:
            task_id (str): The ID of the task.
        """
        self._ensure_task_exists(task_id)
        if task_id not in self.augmentations:
            self.augmentations[task_id] = []
        self.augmentations[task_id].append(strategy)

    # Note this is an example of how to add a complex method to the DataAugmentationManager it may not be strictly needed
    def reverse_augmentations(self, task_id: str, data: np.array) -> str:
        """
        Reverse all augmentations applied to the data for a specific task.

        Args:
            task_id (str): The ID of the task.
            data (str): The augmented data to be reversed.

        Returns:
            str: The data after reversing all augmentations.
        """
        if task_id not in self.augmentations:
            return data
        reversed_data = data
        for strategy in reversed(self.augmentations[task_id]):
            reversed_data = strategy.reverse(reversed_data)
        return reversed_data
