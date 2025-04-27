import random
from typing import List
from .value import Value


class Neuron:
    def __init__(self, n_units: int):
        self.weights =  [Value(random.gauss(0, 1)) for _ in range(n_units)]
        self.bias = Value(random.gauss(0, 1))
    
    def __repr__(self):
        return (
            f"Neuron:\n"
            f"  weights: {self.weights}\n"
            f"  bias: {self.bias}"
        )
    
    def forward(self, x: List[Value]) -> List[Value]:
        assert len(x) == len(self.weights), "Input vector's length must be equal to n_units of the neuron"
        res = Value(0.0)
        for i in range(len(x)):
            res += self.weights[i] * x[i]

        res += self.bias
        return res
    
    def __call__(self, x: List[Value]):
        return self.forward(x)