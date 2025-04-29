from typing import List
from math import e

from .value import Value

def softmax(x: List[Value]):
    exp_list = [e ** (i.data) for i in x]
    exp_sum = sum(exp_list)
    for i in range(len(exp_list)):
        exp_list[i] /= exp_sum
    
    return [Value(i) for i in exp_list]


def argmax(x: List[Value]):
    data_list = [i.data for i in x]
    return Value(data_list.index(max(data_list)))