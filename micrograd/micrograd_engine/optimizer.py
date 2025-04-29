from .neuron_layer import NeuronLayer
from .mlp import MLP


class Optimizer:
    def zero_grad(self):
        for layer in self.mlp.layers:
            if isinstance(layer, NeuronLayer):
                for neuron in layer.layer:
                    neuron.bias.grad = 0.0
                    for weight in neuron.weights:
                        weight.grad = 0.0


class SGD(Optimizer):
    def __init__(self, mlp_model: MLP, lr: float = 0.01):
        self.mlp = mlp_model
        self.lr = lr
    
    def step(self):
        for layer in self.mlp.layers:
            if isinstance(layer, NeuronLayer):
                for neuron in layer.layer:
                    neuron.bias.data -= self.lr * neuron.bias.grad
                    for weight in neuron.weights:
                        weight.data -= self.lr * weight.grad


class Adam(Optimizer):
    def __init__(self, mlp_model: MLP, lr: float = 0.01):
        self.mlp = mlp_model
        self.lr = lr
    
    # under construction...
    
