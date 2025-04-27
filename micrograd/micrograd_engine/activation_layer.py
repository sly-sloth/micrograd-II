from .activation_functions import *


class ActivationLayer:
    def __init__(self, activation: str):
        activations = {
            "tanh": tanh,
            "relu": relu,
            "sigmoid": sigmoid
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.activation = activations[activation]

    def __call__(self, x: List[Value]) -> List[Value]:
        return self.activation(x)