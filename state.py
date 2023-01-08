import numpy as np


class State:
    def __init__(self):
        self.epoch_start = 0
        self.valid_obj_l_min = np.inf

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def print_variables(self):
        for key in self.__dict__:
            print(key, ":", self.__dict__[key])

    def delete_variable(self, variable):
        delattr(self, variable)


# state = State()
# state.update_state(a="1")
# state.print_variables()
# state.delete_variable("a")
# state.print_variables()
