from typing import List
from .value import Value
from .neuron import Neuron


class NeuronLayer:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.layer = [Neuron(in_features) for _ in range(out_features)]
    
    def forward(self, x: List[Value]) -> List[Value]:
        assert len(x) == self.in_features, "Input vector's length must be equal to n_units of the neuron"
        res = []
        for i in range(self.out_features):
            neuron_i = self.layer[i]
            res_i = Value(0.0)
            for j in range(self.in_features):
                res_i += x[j] * neuron_i.weights[j]
            res_i += neuron_i.bias
            res.append(res_i)

        return res
    
    def __call__(self, x: List[Value]):
        return self.forward(x)

    def __repr__(self):
        out_str = "NeuronLayer:\n"
        for i in range(self.out_features):
            neuron = self.layer[i]
            # out_str += f"Neuron_{i+1}\n" + str(neuron) + "\n"
            out_str += str(neuron) + "\n"
        return out_str