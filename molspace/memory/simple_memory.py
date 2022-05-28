class SimpleMemory:

    def __init__(self):
        self.states = []
        self.rewards = []

    def store(self, state, reward=None):
        self.states.append(state)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.states) - 2

    def retrieve(self, i):
        return (
            (self.states[i], self.states[i + 1]),
            self.rewards[i + 1],
            (self.states[i + 1], self.states[i + 2]),
        )
