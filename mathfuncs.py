import objectdefs as od
import random
e = 2.718281828459045
output_list = [0,0,0,0,0,0,0,0,0,0]
def matrix_multiply(m1,m2):
    num_rows_m1, num_cols_m1 = len(m1), len(m1[0])
    num_rows_m2, num_cols_m2 = len(m2), len(m2[0])
    if num_cols_m1 != num_rows_m2:
        print("These cannot be multiplied")
        return
    m3 = []
    for i in range(num_rows_m1):
        m3.append([])
        for j in range(num_cols_m2):
            m3[i].append(0)
            for k in range(num_cols_m1):
                m3[i][j] += m1[i][k] * m2[k][j]
    return m3

def relu(value):
    result = (abs(value)+value)/2.0
    return result

def sigmoid(value, derivative_or_not = "no"):
    result = 1/(1+e**(-value))
    if derivative_or_not == "deriv":
        deriv_result = sigmoid(value)*(1-sigmoid(value)) #this weird equation is just a more efficient way of writing the derivative of e^x/(1+e^x), which ends up just being e^x/((1+e^x)^2)
        return deriv_result
    return result

def proto_cost_function(correct_output, layer, avg_or_not = "avg"):
# assume correct_number is an int 0-9
    actual_activations = layer.get_neuron_values()
    cost_list = []
    for i in range(10):
        cost = pow(actual_activations[i]-correct_output[i],2)
        cost_list.append(cost)
    if avg_or_not == "avg":
        avg_cost = sum(cost_list) / 10
        return avg_cost
    return cost_list

def is_correct(correct_output, output_layer):
    expected = correct_output
    should_be_highest = output_layer.get_neuron_values()[expected]
    for value in output_layer.get_neuron_values():
        if value > should_be_highest:
            return False
    return True
def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        result.append([])
        for j in range(len(matrix)):
            result[i].append(matrix[j][i])
    return result


def find_one_weight_gradient(input_neuron, output_neuron, correct_number):
    specific_weight = input_neuron.weights[output_neuron.number]
    #return f"Neuron {input_neuron.number} of layer {input_neuron.layer.layer_number} is connected to neuron {output_neuron.number} of layer {output_neuron.layer.layer_number} with a weight of {specific_weight}"\
    new_list = output_list.copy()
    new_list[correct_number] = 1
    difference = output_neuron.value - new_list[output_neuron.number]
    result = 2 * difference * sigmoid(input_neuron.value * specific_weight + input_neuron.layer.get_bias_weights()[input_neuron.number], "deriv") * input_neuron.value
    #this comes from the formula for the partial derivative of the cost function of one neuron with respect to the neuron weight+value+bias it originates from
    return result*-1

def add_lists(list1, list2):
    #assumes they are the same size, changes list1
    for i in range(len(list1)):
        list1[i] += list2[i]
    return list1

def recursive_add_lists(list1, list2):
    if not isinstance(list1, list) or not isinstance(list2, list):
        return list1 + list2
    else:
        return [recursive_add_lists(l1, l2) for l1, l2 in zip(list1, list2)]
def hadamard_product(matrix1, matrix2):
    #assumes they are the same size
    result = []
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result.append(matrix1[i][j] * matrix2[i][j])
    return result

def find_weight_gradient(error_matrix, prev_layer):
    #print(f"This is the error matrix of layer {prev_layer.layer_number + 1}: \n {error_matrix}")
    weight_gradient = []
    for i in range(len(error_matrix)):
        weight_gradient.append([])
        for neuron in prev_layer.neurons:
            weight_gradient_component = neuron.value * error_matrix[i]
            # each prev_layer neuron value is multiplied by the same output_error value
            # means that final form of gradient will be (if prev_layer has k neurons, output has j neurons) [[x(0,0), x(0, 1), x(0,2) ... x(0,k)], [x(1,0) ... x(1,k)] ... ... [x(j, 0), x(j, 1) ... x(j,k)]]
            weight_gradient[i].append(weight_gradient_component)
    return weight_gradient

def find_output_layer_error(network, correct_values):
    #cost function is quadratic -- C = 1/2(a-y)^2
    #first find partial derivative of cost function w/ respect to activation = (a-y)
    #will be easier to do this calculation in matrix form
    output_error_list = []
    output_layer = network.layer_list[-1] #returns last layer given network
    for neuron in output_layer.neurons:
        difference = neuron.value - correct_values[neuron.number]
        output_error_list.append(difference * sigmoid(neuron.weighted_input, "deriv"))
        #assumes correct_values is the same size as the output layer of neurons and is in the correct order to match up with those neurons
        #this comes from the formula for the error (partial deriv w/ respect to z) of the output layer
        #at this point, we should have a list containing the error values for each neuron in the output layer
    return output_error_list

def find_error_from_next_layer(layer, next_layer_error):
    weight_matrix = layer.get_neuron_weights()
    weight_matrix = transpose(weight_matrix)
    intermediate_value = matrix_multiply([next_layer_error], weight_matrix)
    layer_weighted_inputs = layer.get_weighted_inputs()
    deriv_sigmoid_weighted_input_matrix = [[]]
    for z in layer_weighted_inputs:
        deriv_sigmoid_weighted_input_matrix[0].append(sigmoid(z, "deriv"))
    layer_error = hadamard_product(intermediate_value, deriv_sigmoid_weighted_input_matrix)
    return layer_error
    # this is all from the formula to find the error matrix of a layer given the error matrix of the next layer

def recursive_const_mult_matrix(constant, matrix):
    if not isinstance(matrix, list):
        return matrix*constant
    else:
        return[recursive_const_mult_matrix(constant, sublist) for sublist in matrix]

def deepcopy(array):
    if not isinstance(array, list):
        return array
    else:
        return [deepcopy(array[i]) for i in range(len(array))]

def read_image(input_layer, image_number):
    file = open(path, mode = "r+b") #path should be set to the path of the image set file
    file.seek(16 + image_number*784) #change back to 16+ ...
    test = []
    for i in range(len(input_layer.neurons)): #change back to 784
        byte = file.read(1)
        value = (int.from_bytes(byte, byteorder='big'))/255
        input_layer.neurons[i].value = sigmoid(value)
        #input_layer.neurons[i].value = 1
def read_label(label_number, return_type = "list"):
    correct_output_list = [0,0,0,0,0,0,0,0,0,0]
    file = open(path, mode = "r+b") #path should be set to the path of the label set file
    file.seek(8+label_number)
    value = int.from_bytes(file.read(1), byteorder='big')
    if return_type == "num":
        return value
    correct_output_list[value] = 1
    return correct_output_list

def to_file(values, path):
    file = open(path, mode = 'w')
    file.write(values)
