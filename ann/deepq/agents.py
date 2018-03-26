from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np, random as rd, math
from pathlib import Path


'''
Q-Learning with Experience Replay
TODO Add a target network to stabilize learning
https://arxiv.org/pdf/1509.02971.pdf
'''
class DeepQAgent:

    def __init__(self, state_num, action_num):
        self.state_num = state_num
        self.action_num = action_num

         # initialize Hyperparameters
        self.memory_capacity = 100000
        self.batch_size = 64
        # Gamma: Discount Factor
        self.gamma = 0.90
        # Epsilon: Greedy Factor
        self.epsilon_max = 1 # EXPLORE EVERYTHING
        self.epsilon_min = 0.01 # 1% Chance for Explore, else Exploit
        self.epsilon = self.epsilon_max
        # Lambda: Greed Factor Decay
        # underscore is there cause lambda is a reserved keyword
        self.lambda_ = 0.001
        self.steps = 0
        self.update_frequency = 1000
        # initialize the brain
        self.brain = DeepQBrain(state_num, action_num, self.memory_capacity)


    def act(self, s):
        # randomly choose any action if less than epsilon
        if rd.random() < self.epsilon:
            return rd.randint(0, self.action_num - 1)
        else:
            return np.argmax(self.brain.predict(s))

    def observe(self, sample): # (s, a, r, s_)
        self.brain.store(sample)

        # update the target model periodically
        if self.steps % self.update_frequency == 0:
            self.brain.update_target()

        # insert debug there
        # if self.steps % 100 == 0:
        # state
        # prediction

        # decrease epsilon
        self.steps += 1
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.lambda_ * self.steps)

    def replay(self):
        recall_entries = self.brain.recall(self.batch_size)
        recall_size = len(recall_entries)
        # create a null state with size state
        null_state = np.zeros(self.state_num)
        # (s, a, r, s_)
        states  = np.array([ recall_entry[0] for recall_entry in recall_entries ])
        states_ = np.array([ ( null_state if recall_entry[3] is None else recall_entry[3] ) for recall_entry in recall_entries ])

        action = self.brain.predict_batch(states)
        action_ = self.brain.predict_batch(states_)

        # zeroing map x to y (state -> actions)
        x = np.zeros((recall_size, self.state_num))
        y = np.zeros((recall_size, self.action_num))

        # use experience for training
        for i in range(recall_size):
            s, a, r, s_  = recall_entries[i]
            # target action
            t = action[i]
            # if the next state is none
            # put the reward, else use the equation??
            # this is for the target (t)
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(action_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

    def save_model(self):
        self.brain.save_model()

# Deep Q Model Implementations with Memory
class DeepQBrain:

    def __init__(self, state_num, action_num, memory_capacity):
        self.state_num = state_num
        self.action_num = action_num
        self.memory_capacity = memory_capacity
        self.memory_entries = []
        self.model_filename = "deepq.h5"

        '''
        TODO make this a function argument?
        separated this hyperparameter because
        this is specific to the model and not the Q Learning
        learning rate
        '''
        self.learning_rate = 0.00001

        # load model from file if found
        # _model is the target model
        model_file = Path(self.model_filename)
        if model_file.exists():
            self.model  = load_model(self.model_filename)
            self.model_ = load_model(self.model_filename)
            print("********** Model loaded from file: ", self.model_filename)
        else:
            self.model  = self._create_model()
            self.model_ = self._create_model()

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, epochs=epochs, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s.reshape(1, self.state_num)).flatten()

    def predict_batch(self, s):
        return self.model.predict(s)

    def save_model(self):
        self.model_.save(self.model_filename)
        print("********** Model saved into file: ", self.model_filename)

    def _create_model(self):
        model = Sequential()
        # input -> Dense[64] -> output
        model.add(Dense(32, kernel_initializer='lecun_uniform', activation='relu', input_dim=self.state_num))
        model.add(Dense(16, kernel_initializer='lecun_uniform', activation='relu'))
        model.add(Dense(self.action_num, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def update_target(self):
        self.model_.set_weights(self.model.get_weights())

    def store(self, memory_entry):
        # forget the oldesst one when beyond capacities
        if len(self.memory_entries) > self.memory_capacity:
            self.memory_entries.pop(0)
        self.memory_entries.append(memory_entry)

    def recall(self, size):
        n = min(size, len(self.memory_entries))
        return rd.sample(self.memory_entries, n)
