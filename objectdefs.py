import random
class Neuron:
    neuron_count = 0
    def __init__(self, value=0, weights=None, number = 0, layer = None):
        self.value = value
        self.number = number
        self.layer = layer
        if weights is None:
            self.weights = []
        else:
            self.weights = weights
        Neuron.neuron_count += 1

    def __str__(self):
        return (f"This neuron has a value of {self.value} and the following list of weights:\n{self.weights}")

    def initialize_weights(self, size):
        self.weights.clear()
        for i in range(size):
            self.weights.append(0)

    def randomize_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = random.random()

    def initialize_value(self, value):
        self.value = value

class Layer:
    layer_count = 0
    def __init__(self, size, layer_connection = None):
        self.neurons = [Neuron(number=i, layer = self) for i in range(size)]
        self.layer_connection = layer_connection
        Layer.layer_count += 1
        self.layer_number = Layer.layer_count
        self.bias = Neuron(0.0001)
        self.number_of_connections = 0

    def __str__(self):
        x = f"originates from layer {self.layer_connection.layer_number} " if self.layer_connection is not None else 'is the first layer '
        str = f"Layer {self.layer_number} {x}and consists of the following neurons:\n"
        i=1
        for neuron in self.neurons:
            str += f"   Neuron {neuron.number} has a value of {neuron.value} and the following list of weights:\n   {neuron.weights}\n"
            i += 1
        return str + f"      Bias weights: {self.bias.weights}\n"

    def randomize_layer_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights()

    def initialize_neuron_connections(self, number_of_connections):
        for neuron in self.neurons:
            neuron.initialize_weights(number_of_connections)
        print(f"The neurons (and bias) in layer {self.layer_number} have now been initialized with weights of {[0 for i in range(number_of_connections)]}")
        self.number_of_connections = number_of_connections
        self.bias.initialize_weights(number_of_connections)

    def initialize_neuron_values(self, list_of_values):
        for neuron, value in zip(self.neurons, list_of_values):
            neuron.value = value

    def randomize_bias_weights(self):
        self.bias.randomize_weights()

    def get_neuron_values(self):
        return [neuron.value for neuron in self.neurons]

    def get_neuron_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def get_bias_weights(self):
        return self.bias.weights

