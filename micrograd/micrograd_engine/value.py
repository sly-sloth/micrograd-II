import math
from typing import List


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self.label = label
        self.visited = set()

    def __repr__(self):
        # if show_grad:
        return f"Value(data={self.data}, grad={self.grad})"
        # return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __sub__(self, other):
        out = Value(self.data.__add__(-1 * other.data), (self, other), '-')
        return out
    # figure if this gets automatically handled or not
    # soln: neural nets don't need subtraction
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    def tanh(self):
        x_data = self.data
        t = math.tanh(x_data)
        out = Value(t, (self,), "tanh")
        return out

    def sigmoid(self):
        x_data = self.data
        t = 1 / (1 + math.exp(-x_data))
        out = Value(t, (self,), "sigmoid")
        return out
    
    def relu(self):
        x_data = self.data
        t = 0 if x_data <= 0 else x_data
        out = Value(t, (self,), "relu")
        return out
    
    def trace_back(self, node):
        if node in self.visited:
            return
        self.visited.add(node)
        
        if node._prev:
            if len(node._prev) > 1:
                if node._op == '+':
                    if node._prev[0] != node._prev[1]:
                        node._prev[0].grad += node.grad * 1.0
                        node._prev[1].grad += node.grad * 1.0
                    else:
                        node._prev[0].grad += node.grad * 1.0
                elif node._op == '-':
                    if node._prev[0] != node._prev[1]:
                        node._prev[0].grad += node.grad * 1.0
                        node._prev[1].grad += node.grad * 1.0
                    else:
                        node._prev[0].grad += node.grad * 1.0
                else:
                    if node._prev[0] != node._prev[1]: # or take a set
                        node._prev[0].grad += node.grad * node._prev[1].data
                        node._prev[1].grad += node.grad * node._prev[0].data
                    else:
                        node._prev[0].grad += node.grad * node._prev[0].data

                node.trace_back(node._prev[0])
                node.trace_back(node._prev[1])
            elif len(node._prev) == 1:
                if node._op == "tanh":
                    # print("tanh op found")
                    node._prev[0].grad += node.grad * (1 - node.data * node.data)
                elif node._op == "relu":
                    # print("relu op found")
                    if node._prev[0].data > 0:
                        node._prev[0].grad += node.grad * 1
                elif node._op == "sigmoid":
                    # print("sigmoid op found")
                    node._prev[0].grad += node.grad * node.data * (1 - node.data)
                else:
                    raise ValueError("Unknown operation found, not supported!")

                node.trace_back(node._prev[0])

    def backward(self):
        self.grad = 1.0
        self.visited = set()
        self.trace_back(self)