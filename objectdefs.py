import random
import mathfuncs as mf
import concurrent.futures as cf
import os
import threading
import multiprocessing
import copy
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
            self.weights[i] += desired_change[i]


    def assign_weighted_input(self, z):
        self.weighted_input = z

    def get_weighted_input(self):
        return self.weighted_input

    def assign_value(self, value):
        self.value = value

    def copy(self):
        return Neuron(self.value, copy.copy(self.weights), self.number, self.layer)

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
        self.number_of_connections = number_of_connections
        self.bias.initialize_weights(number_of_connections)

    def assign_neuron_values(self, list_of_values):
        for neuron, value in zip(self.neurons, list_of_values):
            neuron.value = value

    def assign_weighted_inputs(self, weighted_inputs):
        for neuron, z in zip(self.neurons, weighted_inputs):
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

    def copy(self):
        copy = Network(self.LEARNING_RATE, input_layer=self)

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
            new_layer_values[j] = mf.sigmoid(new_layer_values[j])
        new_layer = Layer(len(new_layer_values), layer)
        new_layer.assign_neuron_values(new_layer_values)
        new_layer.assign_weighted_inputs(new_layer_weighted_inputs)
        return new_layer

    def recursive_backprop(self, correct_values, layer = None):
        if layer is None:
            layer = self.layer_list[0]            
        if(layer==self.layer_list[-1]):
            output_error = mf.find_output_layer_error(self, correct_values)
            self.layer_list[-1].error = output_error
            return output_error
        layer.error = mf.find_error_from_next_layer(layer, self.recursive_backprop(correct_values, self.layer_list[layer.layer_number + 1]))
        layer.weight_gradient = mf.find_weight_gradient(self.layer_list[layer.layer_number+1].error, layer)
        layer.bias_gradient = self.layer_list[layer.layer_number + 1].error
        return layer.error

    def batch_gradient_descent(self, labels, start_num):
        start_weight_grad = []
        start_bias_grad = []
        for layer, i in zip(self.layer_list, range(len(self.layer_list))):
            start_weight_grad.append([])
            start_bias_grad.append(layer.get_bias_weights())
            transpose_weights = mf.transpose(layer.get_neuron_weights())
            for j in range(len(transpose_weights)):
                start_weight_grad[i].append(transpose_weights[j])
        average_weight_gradient = start_weight_grad
        average_bias_gradient = start_bias_grad
        print(f"\nRunning batch_gradient_descent on images {start_num} through {start_num+len(labels)}!\n")
        for i in range(len(labels)):
            mf.read_image(self.layer_list[0], i+start_num)
            self.regenerate_network()
            self.recursive_backprop(labels[i])
            for j in range(len(self.layer_list)):
                new_matrix = mf.recursive_add_lists(average_weight_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].weight_gradient))
                average_weight_gradient[j] = new_matrix
                average_bias_gradient[j] = mf.recursive_add_lists(average_bias_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].bias_gradient))
        for k in range(len(self.layer_list)):
            self.layer_list[k].weight_gradient = average_weight_gradient[k]
            self.layer_list[k].bias_gradient = average_bias_gradient[k]
            self.layer_list[k].update_layer_weights()
            self.layer_list[k].update_bias_weights()


    def multithread_batch_gradient_descent(self, labels, start_num, queue):
        start_weight_grad = []
        start_bias_grad = []
        for layer, i in zip(self.layer_list, range(len(self.layer_list))):
            start_weight_grad.append([])
            start_bias_grad.append(layer.get_bias_weights())
            transpose_weights = mf.transpose(layer.get_neuron_weights())
            for j in range(len(transpose_weights)):
                start_weight_grad[i].append(transpose_weights[j])
        average_weight_gradient = start_weight_grad
        average_bias_gradient = start_bias_grad
        #print(f"\nRunning batch_gradient_descent on images {start_num} through {start_num+len(labels)}!\n")
        for i in range(len(labels)):
            mf.read_image(self.layer_list[0], i+start_num)
            self.regenerate_network()
            self.recursive_backprop(labels[i])
            for j in range(len(self.layer_list)):
                new_matrix = mf.recursive_add_lists(average_weight_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].weight_gradient))
                average_weight_gradient[j] = new_matrix
                average_bias_gradient[j] = mf.recursive_add_lists(average_bias_gradient[j], mf.recursive_const_mult_matrix(1/len(labels), self.layer_list[j].bias_gradient))
        #result =  {'avg_w' : average_weight_gradient, 'avg_b' : average_bias_gradient}
        #queue.put(result)
        '''for k in range(len(self.layer_list)):
            self.layer_list[k].weight_gradient = average_weight_gradient[k]
            self.layer_list[k].bias_gradient = average_bias_gradient[k]
            self.layer_list[k].update_layer_weights()
            self.layer_list[k].update_bias_weights()'''

    '''def test_grad_descent(self, labels, num_cores, start_num):
            process_list = []
            labels_sublists = []
            queue = multiprocessing.Queue()
            for i in range(num_cores):
                current_pos = i*(len(labels)//num_cores)
                labels_sublists.append(labels[current_pos:])
                process_list.append(multiprocessing.Process(target=self.multithread_batch_gradient_descent,
                                                            args = (labels_sublists[i], start_num+current_pos, queue)))
                process_list[i].start()
            for process in process_list:
                process.join()
            for result in queue'''



    def update(self, avg_w, avg_b):
        for k in range(len(self.layer_list)):
            self.layer_list[k].weight_gradient = avg_w[k]
            self.layer_list[k].bias_gradient = avg_b[k]
            self.layer_list[k].update_layer_weights()
            self.layer_list[k].update_bias_weights()





    def regenerate_network(self):
        for layer in self.layer_list[1:]:
            layer.assign_neuron_values(self.generate_next_layer(self.layer_list[layer.layer_number-1]).get_neuron_values())


