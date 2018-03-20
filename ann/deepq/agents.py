class DeepQAgent:
    
    def __init__(self, state_num, action_num):      
    def act(self, s):
    def observe(self, sample): # (s, a, r, s_)       
    def replay(self):    
      
        
class DeepQBrain:
    
    def __init__(self, state_num, action_num):
        self.state_num = state_num
        self.action_num = action_num

        # make this a parameter?
        # learning rate
        self.learning_rate = 0.00001
                
        # load model from file if found
        # self.model.load_weights("cartpole.h5")
        
        self.model = self._createModel()
        

    def _createModel(self):
        model = Sequential()

        # input -> Dense[64] -> output
        model.add(Dense(64, kernel_initializer='lecun_uniform', activation='relu', input_dim=self.state_num))
        model.add(Dense(self.action_num, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.summary()

        return model

    def train(self, x, y, epoch=1, verbose=0):
    def predict(self, s):
    def predictOne(self, s):