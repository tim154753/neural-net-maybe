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
            #self.weights[i] = random.random()
            self.weights[i] = (random.gauss(0,1/len(self.weights)**0.5))

    def update_weights(self, desired_change):
        for i in range(len(self.weights)):
            self.weights[i] += desired_change[i] #this assumes desired_change is the sub-vector of the gradient vector relating to the weights extending out from the neuron in question
            #meaning I'll have to deal with that somewhere else
            #also assumes that our step size is 1 for now, can easily change this though

    def assign_weighted_input(self, z):
        self.weighted_input = z

    def get_weighted_input(self):
        return self.weighted_input

    def assign_value(self, value):
        self.value = value

class Layer:
    layer_count = 0
    def __init__(self, size, layer_connection = None):
        self.neurons = [Neuron(number=i, layer = self) for i in range(size)]
        self.layer_connection = layer_connection
        self.layer_number = Layer.layer_count
        Layer.layer_count += 1
        self.bias = Neuron(0.01)
        self.number_of_connections = 0
        self.error = []
        self.weight_gradient = []
        self.bias_gradient = []
        self.weights = []

    def __str__(self):
        x = f"originates from layer {self.layer_connection.layer_number} " if self.layer_connection is not None else 'is the first layer '
        str = f"Layer {self.layer_number} {x}and consists of the following neurons:\n"
        i=1
        for neuron in self.neurons:
            str += f"   Neuron {neuron.number} has a value of {neuron.value} and the following list of weights:\n   {neuron.weights}\n"
            i += 1
        return str + f"      Bias weights: {self.bias.weights}\n"

    def randomize_layer_weights(self):
        for i in range(len(self.neurons)):
            self.neurons[i].randomize_weights()
            self.weights[i] = self.neurons[i].weights

    def initialize_neuron_connections(self, number_of_connections):
        for neuron in self.neurons:
            neuron.initialize_weights(number_of_connections)
            self.weights.append(neuron.weights)
        #print(f"The neurons (and bias) in layer {self.layer_number} have now been initialized with weights of {[0 for i in range(number_of_connections)]}")
        self.number_of_connections = number_of_connections
        self.bias.initialize_weights(number_of_connections)

    def assign_neuron_values(self, list_of_values):
        for neuron, value in zip(self.neurons, list_of_values):
            neuron.value = value

    def assign_weighted_inputs(self, weighted_inputs):
        for neuron, z in zip(self.neurons, weighted_inputs):
            #print(f"\nAssigning neuron {neuron.number} of layer {self.layer_number} with a weighted input of {z}!")
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

    def update_layer_weights(self):
        for i in range(len(self.neurons)):
            #print(f"From update_layer_weights: The current weight matrix of layer {self.layer_number} is \n{self.get_neuron_weights()}\n")
            #print(f"From update_layer_weights: The current weight gradient matrix of layer {self.layer_number} is \n{self.weight_gradient}\n")
            for j in range(len(self.weight_gradient)):
                step = self.weight_gradient[j][i] * -1 * Network.LEARNING_RATE
                self.neurons[i].weights[j] += step
    def update_bias_weights(self):
        for i in range(len(self.bias.weights)):
            step = self.bias_gradient[i] * -1 * Network.LEARNING_RATE
            self.bias.weights[i] += step

class Network:
    LEARNING_RATE = 0.001
    def __init__(self, learning_rate = 1, input_layer = None, size = 0):
        self.layer_list = []
        self.initial_layer = input_layer
        self.size = size
        Network.LEARNING_RATE = learning_rate

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
            #print(f"Adding {new_layer_values[j]} onto the list of weighted inputs!")
            new_layer_values[j] = mf.sigmoid(new_layer_values[j])
        #print(f"\nThe list of weighted inputs for this new layer is now {new_layer_weighted_inputs}!")
        new_layer = Layer(len(new_layer_values), layer)
        new_layer.assign_neuron_values(new_layer_values)
        new_layer.assign_weighted_inputs(new_layer_weighted_inputs)
        return new_layer

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


    def recursive_backprop(self, correct_values, layer = None):
        if layer is None:
            layer = self.layer_list[0]            
        if(layer==self.layer_list[-1]):
            output_error = mf.find_output_layer_error(self, correct_values)
            self.layer_list[-1].error = output_error
            return output_error
        #print(layer)
        #print(self.layer_list[layer.layer_number + 1])
        #next_layer_error = self.layer_list[layer.layer_number+1].error
        layer.error = mf.find_error_from_next_layer(layer, self.recursive_backprop(correct_values, self.layer_list[layer.layer_number + 1]))
        #print(f"From recursive_backprop: This is layer {layer.layer_number}'s error matrix \n{layer.error}")
        layer.weight_gradient = mf.find_weight_gradient(self.layer_list[layer.layer_number+1].error, layer)
        #print(f"From recursive_backprop: This is layer {layer.layer_number}'s weight gradient: \n {layer.weight_gradient}")
        #print(f"From recursive_backprop: This is layer {layer.layer_number+1}'s error matrix: \n{self.layer_list[layer.layer_number+1].error}")
        layer.bias_gradient = self.layer_list[layer.layer_number + 1].error
        return layer.error

    def batch_gradient_descent(self, labels, start_num):



        average_weight_gradient = []
        average_bias_gradient = []
        weights=[]
        start_weight_grad = []
        start_bias_grad = []
        for layer, i in zip(self.layer_list, range(len(self.layer_list))):
            #average_weight_gradient.append([])
            #average_bias_gradient.append(layer.get_bias_weights())
            start_weight_grad.append([])
            start_bias_grad.append(layer.get_bias_weights())
            transpose_weights = mf.transpose(layer.get_neuron_weights())
            for j in range(len(transpose_weights)):
                #average_weight_gradient[i].append(transpose_weights[j])
                start_weight_grad[i].append(transpose_weights[j])

        average_weight_gradient = start_weight_grad
        average_bias_gradient = start_bias_grad
            #print(f"From batch_gradient_descent: These are the weights of neuron of layer {layer.layer_number}:\n {weights}")
            #average_weight_gradient.append([0 for i in range(len(layer.get_neuron_weights()))])
            #average_bias_gradient.append([0 for i in range(len(layer.get_neuron_weights()))])
        #print(f"From batch_gradient_descent: This is average weight gradient: \n{average_weight_gradient}\n")
        #print(f"From batch_gradient_descent: This is average bias gradient: \n{average_bias_gradient}")
        print(f"\nRunning batch_gradient_descent on images {start_num} through {start_num+len(labels)}!\n")
        for i in range(len(labels)):

            #average_weight_gradient = mf.deepcopy(start_weight_grad)
            #average_bias_gradient = mf.deepcopy(start_bias_grad)
            mf.read_image(self.layer_list[0], i+start_num)
            #print(f"From batch_gradient_descent: This is input neuron values:\n {self.layer_list[0].get_neuron_values()}")
            self.regenerate_network()
            self.recursive_backprop(labels[i])
            for j in range(len(self.layer_list)):
                #print(f"This is layer {j}'s weight matrix: {self.layer_list[j].get_neuron_weights()}")
                #print(f"This is layer {j}'s bias matrix: {self.layer_list[j].bias_gradient}")
                #print(f"This is layer {j}'s weight gradient: {self.layer_list[j].weight_gradient}")

                new_matrix = mf.recursive_add_lists(average_weight_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].weight_gradient))

                #print(f"\n\nNew matrix's length is {len(new_matrix)} with {len(new_matrix[0])} entries per row? and is this:\n{new_matrix}\n\n as compared to layer {j}'s weight gradient, which has len {len(self.layer_list[j].weight_gradient)} with {len(self.layer_list[j].weight_gradient[0])} entries per row? and is:\n{self.layer_list[j].weight_gradient}")

                average_weight_gradient[j] = new_matrix
                average_bias_gradient[j] = mf.recursive_add_lists(average_bias_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].bias_gradient))

                #print(f"This is average_weight_gradient after iteration {i}: {average_weight_gradient}")
                #print(f"This is average_bias_gradient after iteration {i}: {average_bias_gradient}")

                #recursive_add_lists returns a new list

                #something is going wrong with the weight gradients, average_weight_gradient is way too short (not even as long as weight matrix 1)

                #self.layer_list[j].weight_gradient = average_weight_gradient[j]
                #self.layer_list[j].bias_gradient = average_bias_gradient[j]
                #self.layer_list[j].update_layer_weights()
                #self.layer_list[j].update_bias_weights()
                #average_weight_gradient[j] += mf.constant_multiply_matrix(1/len(labels), self.layer_list[j].weight_gradient)
                #average_bias_gradient[j] += mf.constant_multiply_matrix(1/len(labels), [self.layer_list[j].bias_gradient])
        #print(f"\nFrom batch_gradient_descent: This is average weight gradient after iterated backprop:\n{average_weight_gradient}")
        #print(f"From batch_gradient_descent: This is average bias gradient after iterated backprop:\n{average_bias_gradient}")
        for k in range(len(self.layer_list)):
            #list is being passed by reference, probably causing some issues
            self.layer_list[k].weight_gradient = average_weight_gradient[k]
            self.layer_list[k].bias_gradient = average_bias_gradient[k]
            self.layer_list[k].update_layer_weights()
            self.layer_list[k].update_bias_weights()
        #label must be an array of 10 numbers, all zero, with the correct number index corresponding to the image as 1

    def regenerate_network(self):
        #assumes batch_gradient_descent has already been run
        for layer in self.layer_list[1:]:
            layer.assign_neuron_values(self.generate_next_layer(self.layer_list[layer.layer_number-1]).get_neuron_values())


