import random
from collections import namedtuple, deque
import itertools

GOOD_MEMORY_THOLD   = -100
BAD_MEMORY_THOLD    = -200

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
        
        
        partial_buffer, remainder_buffer = divmod(buffer_size, 3)
        
        self.good_memory   = deque(maxlen=int(partial_buffer))
        self.bad_memory    = deque(maxlen=int(partial_buffer))
        self.common_memory = deque(maxlen=int(partial_buffer + remainder_buffer))
        
        self.partial_batch, self.remainder_batch = divmod(self.batch_size, 3)
        self.num_experiences = 0
        
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        e = self.experience(state, action, reward, next_state, done)
        
        
        if self.num_experiences < self.buffer_size:
            self.memory.append(e)
            self.num_experiences += 1
        else:
            print("Overwriting Memory")
            self.memory.popleft()
            self.memory.append(e)
        
        #self.memory.append(e)
        
        #if reward > GOOD_MEMORY_THOLD:
        #    self.good_memory.append(e)
        #elif reward < BAD_MEMORY_THOLD:
        #    self.bad_memory.append(e)
        #else:
        #    self.common_memory.append(e)
        

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        return random.sample(self.memory, k=self.batch_size)
    
        if (
            len(self.good_memory) >= self.partial_batch and
            len(self.bad_memory) >= self.partial_batch  and
            len(self.common_memory) >= self.partial_batch+self.remainder_batch
            ):
            #print("Sampling 3 memories")
            sample_good   = random.sample(self.good_memory, k=self.partial_batch)
            sample_bad    = random.sample(self.bad_memory,  k=self.partial_batch)
            sample_common = random.sample(self.common_memory,  k=self.partial_batch+self.remainder_batch)
            #sample = deque(maxlen=batch_size)
            sample = sample_good
            sample += sample_bad
            sample += sample_common
            
            return random.sample(sample, k=self.batch_size)
        else:
            return random.sample(self.memory, k=self.batch_size)
        

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
