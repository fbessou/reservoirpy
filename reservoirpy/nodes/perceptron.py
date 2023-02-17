from ..node import Node

def perceptron_forward(perceptron: Node, data):
    return perceptron.weight @ data + perceptron.bias

def perceptron_initializer(perceptron: Node, x=None, weight, bias, *args, **kwargs):
    if x is not None:
        node.set_input_dim(x.shape[1])
        
class Perceptron(Node):
    