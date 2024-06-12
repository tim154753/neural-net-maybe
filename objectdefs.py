import random
class Neuron:
    neuron_count = 0
    def __init__(self, value=0, weights=None):
        self.value = value
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
    def __init__(self, size):
        self.neurons = [Neuron() for i in range(size)]
        Layer.layer_count += 1
        self.layer_number = Layer.layer_count

    def __str__(self):
        str = f"Layer {self.layer_number} consists of the following neurons:\n"
        i=1
        for neuron in self.neurons:
            str += f"   Neuron {i} has a value of {neuron.value} and the following list of weights:\n   {neuron.weights}\n"
            i += 1
        return str

    def randomize_layer_weights(self):
        for neuron in self.neurons:
            neuron.randomize_weights()

    def initialize_neuron_connections(self, number_of_connections):
        for neuron in self.neurons:
            neuron.initialize_weights(number_of_connections)
        print(f"The neurons in layer {self.layer_number} have now been initialized with weights of {[0 for i in range(number_of_connections)]}")

    def initialize_neuron_values(self, list_of_values):
        for neuron, value in zip(self.neurons, list_of_values[0]):
            neuron.value = value

    def get_neuron_values(self):
        result  = []
        for neuron in self.neurons:
            result.append(neuron.value)
        return result
    def get_neuron_weights(self):
        result = []
        for neuron in self.neurons:
            result.append(neuron.weights)
        return result
