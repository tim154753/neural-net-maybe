#first, algorithm to multiply matrices
#ex m1 = [[1,2,3],  m2 = [[9,3],
#         [4,5,6],        [8,4],
#         [7,8,9]]        [7,5]]

#make sure len(each row) of m1 = len(column) of m2 = len(m2)
import objectdefs as od
import math

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

def sigmoid(value):
    result = math.e**value
    result = result/(1+math.e**value)
    return result

def generate_next_layer(layer):
    first_layer_values = [layer.get_neuron_values()]
    weights = layer.get_neuron_weights()
    second_layer_values = matrix_multiply(first_layer_values, weights)
    for i in range(len(second_layer_values[0])):
        second_layer_values[0][i] = sigmoid(second_layer_values[0][i])
    second_layer = od.Layer(len(second_layer_values[0]))
    second_layer.initialize_neuron_values(second_layer_values)
    return second_layer

def proto_cost_function(correct_number, layer, avg_or_not = "avg"):
# assume expected_number is an int 0-9
    actual_activations = layer.get_neuron_values()
    expected_activation = [0,0,0,0,0,0,0,0,0,0]
    expected_activation[correct_number] = 1
    cost_list = []
    for i in range(10):
        cost = pow(actual_activations[i]-expected_activation[i],2)
        cost_list.append(cost)
    if avg_or_not == "avg":
        avg_cost = sum(cost_list) / 10
        return avg_cost
    return cost_list
