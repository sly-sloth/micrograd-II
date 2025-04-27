from typing import List
from .value import Value

# activation functions
def tanh(x: List[Value]) -> List[Value]:
    return [i.tanh() for i in x]

def relu(x: List[Value]) -> List[Value]:
    return [i.relu() for i in x]

def sigmoid(x: List[Value]) -> List[Value]:
    return [i.sigmoid() for i in x]