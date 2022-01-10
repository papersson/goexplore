from copy import deepcopy


class Cell:
    def __init__(self, simulator_state, action_history, score):
        self.visits = 0
        self.done = False
        self.update(simulator_state, action_history, score)

    def update(self, simulator_state, action_history, score):
        self.simulator_state = simulator_state
        self.action_history = action_history
        self.score = score

    def increment_visits(self):
        self.visits += 1

    def restore_state(self, env):
        env.unwrapped.restore_state(self.simulator_state)

    def history(self):
        return deepcopy(self.action_history), deepcopy(self.score)

    def is_worse(self, score, actions_taken_length):
        return ((score > self.score)
                or (score == self.score and actions_taken_length < len(self.action_history)))

    def set_done(self):
        self.done = True

    def __repr__(self):
        return f'Cell(score={self.score}, traj_len={len(self.action_history)}, visits={self.visits}, done={self.done})'

    # Make sortable
    def __eq__(self, other):
        return self.score == other.score and self.lenght == other.length

    def __lt__(self, other):
        return (-self.score, len(self.action_history)) < (-other.score, len(other.action_history))
