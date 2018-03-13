from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from abc import ABC

class Agent(ABC):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

class ANN(Agent):
    # Naive Model
    def build_model(self):
        model = Sequential()
        model.add(Dense(4, kernel_initializer='lecun_uniform', input_dim=self.state_size, activation='relu'))
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation='relu'))
        model.add(Dense(4, kernel_initializer='lecun_uniform', activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop())
        model.summary()
        return model


class DeepSARSA(Agent):
    # Deep SARSA Model
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model


class DeepQ(Agent):
    # Deep Q Mountain Car
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

class REINFORCE(Agent):
    # REINFORCE Model
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model
