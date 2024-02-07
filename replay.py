import torch
import numpy as np

class ReplayBuffer:
    """Replay buffer for continual learning using PyTorch."""

    def __init__(self, args, num_tasks, num_classes_per_task, num_samples_per_class):
        """
        Initializes the replay buffer.

        Args:
          num_tasks: Total number of tasks.
          num_classes_per_task: Number of classes per task.
          num_samples_per_class: Number of samples to store per class in the buffer.
          input_shape: Shape of the input data, e.g., (3, 32, 32) for CIFAR-10 images.
        """
        self.num_tasks = args.num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.num_samples_per_class = num_samples_per_class
        self.num_samples_per_task = self.num_classes_per_task * self.num_samples_per_class
        self.input_shape = args.input_size
        self._buffer_size = self.num_tasks * self.num_samples_per_task
        # Initialize tensors for storing images and labels
        self.data = {
            'image': torch.zeros((self._buffer_size, *args.input_size), dtype=torch.float),
            'label': torch.zeros((self._buffer_size,), dtype=torch.long)
        }
        self.cursor = 0  # Current position in the buffer to add new data
        self.old_task_boundary = 0  # Boundary index for the old and new tasks in the buffer
        self.index_dict_list = []  # Stores indices for each task

    def add_example(self, task_id, indices, images, labels):
        """
        Adds examples to the buffer.

        Args:
          task_id: The current task ID.
          indices: Indices in the batch to add to the buffer.
          images: Batch of images.
          labels: Batch of labels.
        """
        for index in indices:
            self.data['image'][self.cursor] = images[index]
            self.data['label'][self.cursor] = labels[index]
            self.cursor += 1

    def get_random_batch(self, batch_size):
        """
        Returns a random batch of data from the buffer.

        Args:
          batch_size: The size of the batch to retrieve.

        Returns:
          A batch of data including images and labels.
        """
        # Ensure sampling is within the current cursor for the buffer
        indices = torch.randint(0, self.cursor, (batch_size,))
        return {'image': self.data['image'][indices], 'label': self.data['label'][indices]}
