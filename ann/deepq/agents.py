from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Q-Learning with Experience Replay
# TODO Add a target network to stabilize learning
# https://arxiv.org/pdf/1509.02971.pdf
class DeepQAgent:
    
    # Hyperparameters
    MEMORY_CAPACITY = 100000
    BATCH_SIZE = 64
    # Gamma: Discount Factor
    GAMMA = 0.90
    # Epsilon: Greedy Factor
    EPSILON_MAX = 1 # EXPLORE EVERYTHING
    EPSILON_MIN = 0.01 # 1% Chance for Explore, else Exploit
    # Lambda: Greed Factor Decay
    LAMBDA = 0.001
    
    
    def __init__(self, state_num, action_num):
        self.state_num = state_num
        self.action_num = action_num
        # initialize the brain
        self.brain = DeepQBrain(state_num, action_num, MEMORY_CAPACITY)
        # initialize Hyperparameters Valiables ONLY
        self.epsilon = EPSILON_MAX
        self.steps = 0
        
    def act(self, s):
        # randomly choose any action if less than epsilon
        if random.random() < self.epsilon:
            return random.randint(0, self.action_num - 1)
        else:
            return numpy.argmax(self.brain.predict())
        
    def observe(self, sample): # (s, a, r, s_)       
        self.brain.remember(sample)
        
        # decrease epsilon
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        
    def replay(self):
        # TODO
      
        
# Deep Q Model Implementations with Memory
class DeepQBrain:
    # Separated this hyperparameted because this is specific to the
    # model and not the Q Learning
    LEARNING_RATE = 0.00001
    
    def __init__(self, state_num, action_num, memory_capacity):
        self.state_num = state_num
        self.action_num = action_num
        self.memory_capacity = memory_capacity
        self.memory_entries = []        
        
        # make this a parameter?
        # learning rate
        self.learning_rate = LEARNING_RATE
                
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
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.summary()

        return model
        
    def remember(memory_entry):
        # forget the oldesst one when beyond capacitys
        if len(self.memory_entries) > self.memory_capacity:
            self.memory_entries.pop(0)
            
        self.memory_entries.append(memory_entry)        
        
        
    def recall(size):
        n = min(size, len(self.memory_entries))
        return random.sample(self.memory_entries, size)
        
