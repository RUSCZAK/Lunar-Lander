import random
from collections import namedtuple, deque
import itertools


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        #print(e)
        self.memory.append(e)
        

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        #sorted_by_reward = sorted(self.memory, key=lambda tup: -tup.reward)
        #deque_slice = deque(itertools.islice(sorted_by_reward, 0, self.batch_size))
        #print(deque_slice)
        return random.sample(self.memory, k=self.batch_size)
        #return random.sample(deque_slice, k=self.batch_size)
        #return deque_slice

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
