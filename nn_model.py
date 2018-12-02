from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.initializers import glorot_normal
import numpy as np

class NNModel(): 
    @staticmethod
    def build(hidden_units, input_dimension):
        model = Sequential()

        model.add(Dense(units = hidden_units[0], activation = 'relu', kernel_initializer = glorot_normal(),
            input_dim = input_dimension))
        
        for i in range(1, hidden_units.size):
            model.add(Dense(units = hidden_units[i], activation = 'relu', 
            kernel_initializer = glorot_normal()))

        model.add(Dense(units = 2))    

        model.compile(loss='mse', optimizer=Adam(lr=0.00001), metrics=["mean_absolute_error"])

        return model

    # continuous = (high_no - low_no + 1) ^ (depth - cur_layer - 1)
    # size = (high_no - low_no + 1) ^ depth 
    @staticmethod
    def generate_combination_low_high(depth, low_no, high_no):
        comb_list = []
        size = (high_no - low_no + 1) ** depth
        size_i = 0

        for i in range(0, depth):
            no_iterations = (high_no - low_no + 1) ** (depth - i - 1)
            cur_col = []
            cur_numb = low_no
            size_i = 0

            while (size_i < size):
                for k in range(0, no_iterations):
                    cur_col.append(cur_numb)
                size_i += no_iterations
                cur_numb += 1
                if (cur_numb > high_no):
                    cur_numb = low_no

            comb_list.append(cur_col)

        combs = np.array(comb_list[0]).reshape((len(comb_list[0]), 1))

        for i in range(1, depth):
            cur_combs = np.array(comb_list[i]).reshape((len(comb_list[i]), 1))

            combs = np.hstack((combs, cur_combs))

        return combs

    # size = low_high_arr[0] * low_high_arr[0] ... low_hig_arr[low_high_arr.size - 1]
    # n * 2
    # no_iter[0] = (high[1]-low[1]) * ... * (low[low.size] - low[low.size])
    @staticmethod
    def generate_combination_low_high_different(low_high_arr):
        depth = low_high_arr.shape[0]
        comb_list = []

        size = low_high_arr[0, 1] - low_high_arr[0, 0] + 1
        for i in range(1, low_high_arr.shape[0]):
            size *= (low_high_arr[i, 1] - low_high_arr[i, 0] + 1)
        size_i = 0
        
        for i in range(0, depth):
            no_iterations = 1
            for k in range(i + 1, depth):
                no_iterations *= (low_high_arr[k, 1] - low_high_arr[k, 0] + 1)

            cur_col = []
            cur_numb = low_high_arr[i, 0]
            size_i = 0

            while (size_i < size):
                for k in range(0, no_iterations):
                    cur_col.append(cur_numb)
                size_i += no_iterations
                cur_numb += 1
                if (cur_numb > low_high_arr[i, 1]):
                    cur_numb = low_high_arr[i, 0]

            comb_list.append(cur_col)

        combs = np.array(comb_list[0]).reshape((len(comb_list[0]), 1))

        for i in range(1, depth):
            cur_combs = np.array(comb_list[i]).reshape((len(comb_list[i]), 1))

            combs = np.hstack((combs, cur_combs))

        return combs