from typing import List
from .value import Value

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

# def CrossEntropyLoss(y_pred: List[Value], )
