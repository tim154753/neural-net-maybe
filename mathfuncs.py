#first, algorithm to multiply matrices
#ex m1 = [[1,2,3],  m2 = [[9,3],
#         [4,5,6],        [8,4],
#         [7,8,9]]        [7,5]]

#make sure len(each row) of m1 = len(column) of m2 = len(m2)
import objectdefs as od

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
    result = e**value
    result = result/(1+e**value)
    if derivative_or_not == "deriv":
        deriv_result = sigmoid(value)*(1/1+e**value) #this weird equation is just a more efficient way of writing the derivative of e^x/(1+e^x), which ends up just being e^x/((1+e^x)^2)
        return deriv_result
    return result

def generate_next_layer(layer):
    first_layer_values = [layer.get_neuron_values()]
    weights = layer.get_neuron_weights()
    new_layer_values = matrix_multiply(first_layer_values, weights)[0]
    for j in range(len(new_layer_values)):
        new_layer_values[j] += layer.get_bias_weights()[j]
    for i in range(len(new_layer_values)):
        new_layer_values[i] = sigmoid(new_layer_values[i])
    new_layer = od.Layer(len(new_layer_values),layer)
    new_layer.initialize_neuron_values(new_layer_values)
    return new_layer

def proto_cost_function(correct_number, layer, avg_or_not = "avg"):
# assume correct_number is an int 0-9
    actual_activations = layer.get_neuron_values()
    expected_activation = output_list
    expected_activation[correct_number] = 1
    cost_list = []
    for i in range(10):
        cost = pow(actual_activations[i]-expected_activation[i],2)
        cost_list.append(cost)
    if avg_or_not == "avg":
        avg_cost = sum(cost_list) / 10
        return avg_cost
    return cost_list

def find_one_weight_gradient(input_neuron, output_neuron, correct_number):
    specific_weight = input_neuron.weights[output_neuron.number]
    #return f"Neuron {input_neuron.number} of layer {input_neuron.layer.layer_number} is connected to neuron {output_neuron.number} of layer {output_neuron.layer.layer_number} with a weight of {specific_weight}"\
    new_list = output_list.copy()
    new_list[correct_number] = 1
    difference = output_neuron.value - new_list[output_neuron.number]
    result = 2 * difference * sigmoid(input_neuron.value * specific_weight + input_neuron.layer.get_bias_weights()[input_neuron.number], "deriv") * input_neuron.value
    #this comes from the formula for the partial derivative of the cost function of one neuron with respect to the neuron weight+value+bias it originates from
    return result*-1

def find_all_gradients(input_layer, output_layer, correct_number):
    for neuron in input_layer.neurons:







#before making a gradient descent function I think i need to make a function to find a specific neuron's desired
    #change in the weights in order to minimize its specific cost
    #function definitely needs a neuron argument, might need a cost argument, not sure if it needs a layer one


#def find_gradient(neurons, weights):
# given a list of neuron values and weights, we should be able to find the gradient function
# the first layer should be easy, but the second layer relies on the first
# this should probably be split into multiple functions




#def hill_climb(idk):
    #what even goes here as an argument?
    #goal: minimize cost function by changing weights between neurons
    #cost function value will just be a number between 0-1
    #cost function variables will be the weights, coefficients will be value of neurons
    #I guess the simplest version of hill_climb would select a random value to change each parameter by, and check if the cost function improved
    #eg if changing weight x decreased the cost function, we would keep moving that way until it started increasing
    #then change next weight, keep changing until that starts increasing, repeat
    #not sure how well this would work in 1000+ dimensions though
    #the weights should probably just be stored in a list
