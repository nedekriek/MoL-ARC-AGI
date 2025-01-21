import numpy as np
from typing import List, Dict

from mol_arc_agi.data_augmentations import AugmentationStrategy, TransposeAugmentation, Rotate90DegreesAugmentation, DataAugmentationManager

class Dataset:
    def __init__(self):
        # Dict format task_id: data
        self.original_data: Dict[str, np.array] = {}
        # Dict format task_id: Dict format sequence_id: augmented_data 
        # Note sequence_id is suggested to be used to track multiple sequences of augmentations being applied to the original task e.g 1. transpose, rotate 2. rotate, rotate ...
        self.augmented_data: Dict[str, Dict[str, np.array]] = {}
        self.augmenter = DataAugmentationManager()

    # note intended to be used with a function that loads the json and formats it as a numpy array
    def add_task(self, task_id: str, data: np.array) -> None:
        self.original_data[task_id] = data
        if 'sequence_id' not in self.augmented_data[task_id]:
            return data
        data = self.augmented_data[task_id]['sequence_id']

    def apply_augmentations(self, task_id: str, strategies: List[AugmentationStrategy], sequence_id: str) -> None:
        data = self.original_data[task_id]
        augmented_data = data
        for strategy in strategies:
            augmented_data = self.augmenter.apply_augmentation(task_id, augmented_data, strategy)
        self.augmented_data[task_id][sequence_id] = augmented_data

    def get_original_data(self, task_id: str) -> str:
        return self.original_data.get(task_id, "")

    def get_augmented_data(self, task_id: str, sequence_id: str) -> str:
        return self.augmented_data.get(task_id, {}).get(sequence_id, "")

    def get_all_data_for_training(self) -> List[Dict[str, str]]:
        all_data = []
        for task_id, original_data in self.original_data.items():
            all_data.append({'task_id': task_id, 'data': original_data, 'type': 'original'})
            for sequence_id, augmented_data in self.augmented_data[task_id].items():
                all_data.append({'task_id': task_id, 'data': augmented_data, 'type': 'augmented', 'sequence_id': sequence_id})
        return all_data
    
    #TODO add method to permute dataset rather than data

# Example usage
dataset = Dataset()
dataset.add_task('task1', 'example data 1')
dataset.add_task('task2', 'example data 2')
dataset.add_task('task3', 'example data 3')

transpose_augmentation = TransposeAugmentation()
rotate_augmentation = Rotate90DegreesAugmentation()

strategies1 = [transpose_augmentation, rotate_augmentation]
strategies2 = [rotate_augmentation, rotate_augmentation]

dataset.apply_augmentations('task1', strategies1, 'sequence1')
dataset.apply_augmentations('task1', strategies2, 'sequence2')

all_data = dataset.get_all_data_for_training()
for data in all_data:
    print(data)
    
# Output:
# {'task_id': 'task1', 'data': 'example data 1', 'type': 'original'}
# {'task_id': 'task1', 'data': 'example data 1 transposed rotated', 'type': 'augmented', 'sequence_id': 'sequence1'}
# {'task_id': 'task1', 'data': 'example data 1 rotated transposed', 'type': 'augmented', 'sequence_id': 'sequence2'}
# {'task_id': 'task2', 'data': 'example data 2', 'type': 'original'}
# {'task_id': 'task3', 'data': 'example data 3', 'type': 'original'}