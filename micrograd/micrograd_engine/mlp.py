from typing import List
from .neuron_layer import NeuronLayer
from .activation_layer import ActivationLayer
from .value import Value


class MLP:
    def __init__(self):
        self.layers: List[NeuronLayer] = []
    
    def add_layer(self, in_features: int, out_features: int):
        layer = NeuronLayer(in_features=in_features, out_features=out_features)
        self.layers.append(layer)
    
    def add_activation_layer(self, activation_layer: ActivationLayer):
        self.layers.append(activation_layer)

    def forward(self, x: List[Value]) -> List[Value]:
        out_l = x
        for i in range(len(self.layers)):
            curr_layer = self.layers[i]
            out_l = curr_layer(out_l)
        
        return out_l
    
    def __call__(self, x: List[Value]):
        return self.forward(x)
    
    def __repr__(self):
        out_str = "MLP:\n"
        for i in range(len(self.layers)):
            layer = self.layers[i]
            out_str += str(layer) + "\n"
        return out_str