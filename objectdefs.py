import random
import mathfuncs as mf
class Neuron:
    neuron_count = 0
    def __init__(self, value=0, weights=None, number = 0, layer = None):
        self.value = value
        self.number = number
        self.layer = layer
        self.weighted_input = 0
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

    def update_weights(self, desired_change):
        for i in range(len(self.weights)):
            self.weights[i] += desired_change[i] #this assumes desired_change is the sub-vector of the gradient vector relating to the weights extending out from the neuron in question
            #meaning I'll have to deal with that somewhere else
            #also assumes that our step size is 1 for now, can easily change this though

    def assign_weighted_input(self, z):
        self.weighted_input = z

    def get_weighted_input(self):
        return self.weighted_input

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
        self.error = []
        self.weight_gradient = []
        self.bias_gradient = []

    def __str__(self):
        x = f"originates from layer {self.layer_connection.layer_number-1} " if self.layer_connection is not None else 'is the first layer '
        str = f"Layer {self.layer_number-1} {x}and consists of the following neurons:\n"
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

    def assign_weighted_inputs(self, weighted_inputs):
        for neuron, z in zip(self.neurons, weighted_inputs):
            print(f"\nAssigning neuron {neuron.number} of layer {self.layer_number} with a weighted input of {z}!")
            neuron.assign_weighted_input(z)

    def randomize_bias_weights(self):
        self.bias.randomize_weights()

    def get_neuron_values(self):
        return [neuron.value for neuron in self.neurons]

    def get_neuron_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def get_bias_weights(self):
        return self.bias.weights

    def get_weighted_inputs(self):
        result = []
        for neuron in self.neurons:
            result.append(neuron.get_weighted_input())
        return result

    def assign_error(self, error):
        self.error.append(error)

class Network:
    def __init__(self, input_layer = None, size = 0):
        self.layer_list = []
        self.initial_layer = input_layer
        self.size = size

    def __str__(self):
        result =  f"This network has {self.size} layers."
        for layer in self.layer_list:
            result += f"\n {layer} --Weight Gradient = {layer.weight_gradient}\n --Bias Gradient = {layer.bias_gradient}"
        return result

    def generate_first_layer(self, size):
        first_layer = Layer(size)
        self.layer_list.append(first_layer)

    def add_layer(self, layer):
        self.layer_list.append(layer)

    def generate_next_layer(self, layer):
        first_layer_values = [layer.get_neuron_values()]
        weights = layer.get_neuron_weights()
        new_layer_values = mf.matrix_multiply(first_layer_values, weights)[0]
        new_layer_weighted_inputs = []
        for j in range(len(new_layer_values)):
            new_layer_values[j] += layer.get_bias_weights()[j]
            new_layer_weighted_inputs.append(new_layer_values[j])
            print(f"Adding {new_layer_values[j]} onto the list of weighted inputs!")
            new_layer_values[j] = mf.sigmoid(new_layer_values[j])
        print(f"\nThe list of weighted inputs for this new layer is now {new_layer_weighted_inputs}!")
        new_layer = Layer(len(new_layer_values), layer)
        new_layer.initialize_neuron_values(new_layer_values)
        new_layer.assign_weighted_inputs(new_layer_weighted_inputs)
        self.layer_list.append(new_layer)

    def backprop(self, correct_values):
        output_layer = self.layer_list[len(self.layer_list) - 1]
        prev_layer = self.layer_list[len(self.layer_list) - 2]
        output_layer_error = mf.find_output_layer_error(self, correct_values)
        output_layer.assign_error(output_layer_error)
        output_layer_weight_gradient = mf.find_weight_gradient(output_layer_error, prev_layer)
        output_layer.weight_gradient = output_layer_weight_gradient
        output_layer.bias_gradient = output_layer_error
        for i in range(len(self.layer_list) - 2, -1, -1):
            print(f"Finding gradient for layer {i}")
            print(f"Layer {i}'s weighted inputs are: \n{self.layer_list[i].get_weighted_inputs()}")
            self.layer_list[i].assign_error(mf.find_error_from_next_layer(self, self.layer_list[i], self.layer_list[i+1].error))
            self.layer_list[i].weight_gradient = mf.find_weight_gradient(self.layer_list[i].error, self.layer_list[i-1])
            self.layer_list[i].bias_gradient = self.layer_list[i].error[0]
        #at this point, each layer should have its weight and bias gradients set

    def train_network(self, number_of_iterations, correct_values):
        for i in range(number_of_iterations):
            self.backprop(correct_values)





