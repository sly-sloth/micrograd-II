from math import log
from typing import List

from .value import Value
from .utils import *

# loss functions
def MSELoss(y_pred: List[Value], y_target: List[Value]) -> Value:
    assert len(y_pred) == len(y_target), "Length of y_pred should be equal to that of y_target"
    loss = Value(0.0)
    m = len(y_pred)
    for i in range(len(y_pred)):
        diff = y_pred[i] - y_target[i]
        loss += diff * diff
    
    # loss.data /= 2 * m
    # TODO: To add this normalization later, right now we can work without it, just to test grad desc.
    return loss

def CrossEntropyLoss(y_logits: List[List[Value]], y_target: List[Value]) -> Value:
    """
    Multicategorical cross entropy loss

    Expects logits as inputs with auto-conversion to class index.
    """
    assert len(y_logits) == len(y_target), "Length of y_pred should be equal to that of y_target"
    y_prob = [softmax(i) for i in y_logits]

    """
    TODO: need to add graphing for values under softmax operation like activations and also change the loss value calculation. otherwise the backward method won't propagate grads due to no chaining present.
    """

    # TODO:
    # add support for direct operations on Value class 
    # could be easy if we subclass float or something similar
    loss = 0.0
    for i in range(len(y_prob)):
        loss += -1 * log(y_prob[i][int(y_target[i].data)].data)

    loss /= len(y_prob)

    # TODO: Add a normalization factor here as well.
    return Value(loss)



