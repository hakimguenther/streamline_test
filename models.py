import numpy as np

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

class Agent(object):
    def __init__(self, env) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def get_action(self, state):
        pass
    
    def update(self, prev_state, action, reward, next_state, done):
        pass


class Q_Learning(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.exploration_rate = 0.33
        self.learning_rate = 0.1
        self.discount_factor = 0.99

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)
    
    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)

        self.q_table[prev_state[0], prev_state[1], action] = new_value


class Sarsa(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.exploration_rate = 0.33
        self.learning_rate = 0.1
        self.discount_factor = 0.99

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)
    
    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        if np.random.uniform(0, 1) < self.exploration_rate:
            next_value = self.q_table[next_state[0], next_state[1], self.action_space.sample()]  # Explore
        else:
            next_value = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_value - old_value)

        self.q_table[prev_state[0], prev_state[1], action] = new_value


class Q_Learning_Adaptive_Exploration(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.visits = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration = 0.1

    def get_action(self, state):
        exploration_terms = np.sqrt(np.divide(1, self.visits[state[0], state[1]]) * np.log(sum(self.visits[state[0], state[1]])))
        np.nan_to_num(exploration_terms, False, np.inf)

        combined_tables = np.add(self.q_table[state[0], state[1]], self.exploration * exploration_terms)

        return np.argmax(combined_tables)
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)

        self.update_visits(prev_state, action)
    
    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)

        self.q_table[prev_state[0], prev_state[1], action] = new_value


    def update_visits(self, state, action):
        self.visits[state[0], state[1], action] += 1

class Q_Learning_Adaptive_Exploration_Softmax(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.visits = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration = 0.1

    def get_action(self, state):
        exploration_terms = np.sqrt(np.divide(1, self.visits[state[0], state[1]]) * np.log(sum(self.visits[state[0], state[1]])))
        np.nan_to_num(exploration_terms, False, np.inf)

        combined_tables = np.add(self.q_table[state[0], state[1]], self.exploration * exploration_terms)

        return np.argmax(combined_tables)
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)

        self.update_visits(prev_state, action)
    
    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)

        self.q_table[prev_state[0], prev_state[1], action] = new_value


    def update_visits(self, state, action):
        self.visits[state[0], state[1], action] += 1

    
class Q_Learning_Eligibility_Traces(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.epoch_vists = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.exploration_rate = 0.33 #epsilon
        self.learning_rate = 0.1 #alpha
        self.discount_factor = 0.99 #gamma
        self.visit_decay = 0.9 #lambda

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)

        self.update_visits(prev_state, action)

        if done:
            self.reset_visits()
    
    def reset_visits(self):
        self.epoch_vists = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        next_value = self.q_table[next_state[0], next_state[1], self.get_action(next_state)]

        temporal_difference = reward + self.discount_factor * next_value - old_value

        for y in range(self.q_table.shape[0]):
            for x in range(self.q_table.shape[1]):
                for a in range(self.q_table.shape[2]):
                    self.q_table[y,x,a] = self.q_table[y,x,a] + self.learning_rate * temporal_difference * self.epoch_vists[y,x,a]


    def update_visits(self, state, action):
        self.epoch_vists[state[0], state[1], action] += 1

        self.epoch_vists = self.discount_factor * self.visit_decay * self.epoch_vists


class Q_Learning_Eligibility_Traces_Adaptive_Exploration(Agent):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.q_table = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.epoch_vists = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

        self.exploration_rate = 0.33 #epsilon
        self.learning_rate = 0.1 #alpha
        self.discount_factor = 0.99 #gamma
        self.visit_decay = 0.9 #lambda

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return self.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit
    
    def update(self, prev_state, action, reward, next_state, done):
        self.update_q_value(prev_state, action, reward, next_state, done)

        self.update_visits(prev_state, action)

        if done:
            self.reset_visits()
    
    def reset_visits(self):
        self.epoch_vists = np.zeros([space.n for space in self.observation_space.spaces]+[self.action_space.n])

    def update_q_value(self, prev_state, action, reward, next_state, done):
        old_value = self.q_table[prev_state[0], prev_state[1], action]

        next_value = self.q_table[next_state[0], next_state[1], self.get_action(next_state)]
        
        temporal_difference = reward + self.discount_factor * next_value - old_value

        for y in range(self.q_table.shape[0]):
            for x in range(self.q_table.shape[1]):
                for a in range(self.q_table.shape[2]):
                    self.q_table[y,x,a] = self.q_table[y,x,a] + self.learning_rate * temporal_difference * self.epoch_vists[y,x,a]


    def update_visits(self, state, action):
        self.epoch_vists[state[0], state[1], action] += 1

        self.epoch_vists = self.discount_factor * self.visit_decay * self.epoch_vists