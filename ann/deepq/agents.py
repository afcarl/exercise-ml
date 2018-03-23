from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np, random as rd, math

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
        
        # initialize the brain
        self.brain = DeepQBrain(state_num, action_num, self.memory_capacity)
       
        
    def act(self, s):
        # randomly choose any action if less than epsilon
        if rd.random() < self.epsilon:
            return rd.randint(0, self.action_num - 1)
        else:
            return np.argmax(self.brain.predict())
        
    def observe(self, sample): # (s, a, r, s_)       
        self.brain.remember(sample)
        
        # decrease epsilon
        # TODO maybe put the steps in the environment?
        self.steps += 1
        self.epsilon.current = self.epsilon.min + (self.epsilon.max - self.epsilon.min) * math.exp(-self.lambda_ * self.steps)
        
    def replay(self):
        recall_entries = self.brain.recall(self.batch_size)
        recall_size = len(recall_entries)
        # create a null state with size state
        null_state = np.zeros(self.state_count)
        # (s, a, r, s_)  
        states  = np.array([ recall_entry[0] for recall_entry in recall_entries ])
        states_ = np.array([ ( null_state if recall_entry[3] is None else recall_entry[3] ) for recall_entry in recall_entries ])
        
        action = self.brain.predictBatch(states)
        action_ = self.brain.predictBatch(states_)
        
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
      
        
# Deep Q Model Implementations with Memory
class DeepQBrain:

    def __init__(self, state_num, action_num, memory_capacity):
        self.state_num = state_num
        self.action_num = action_num
        self.memory_capacity = memory_capacity
        self.memory_entries = []        
        
        '''
        TODO make this a function argument?
        separated this hyperparameter because 
        this is specific to the model and not the Q Learning
        learning rate
        '''
        self.learning_rate = 0.00001
                
        # load model from file if found
        # self.model.load_weights("cartpole.h5")        
        self.model = self._createModel()  

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, epochs=epochs, verbose=verbose)
        
    def predict(self, s):
        return self.model.predict(s.reshape(1, self.state_num)).flatten()
    
    def predictBatch(self, s):
        return self.model.predict(s)
        
    def _createModel(self):
        model = Sequential()
        
        # input -> Dense[64] -> output
        model.add(Dense(64, kernel_initializer='lecun_uniform', activation='relu', input_dim=self.state_num))
        model.add(Dense(self.action_num, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model
        
    def store(self, memory_entry):
        # forget the oldesst one when beyond capacities
        if len(self.memory_entries) > self.memory_capacity:
            self.memory_entries.pop(0)
            
        self.memory_entries.append(memory_entry)        
        
        
    def recall(self, size):
        n = min(size, len(self.memory_entries))
        return random.sample(self.memory_entries, size)
        
