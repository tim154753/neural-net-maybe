import mathfuncs as mf
import objectdefs as od
import random

print("New execution\n")

network = od.Network(size = 3)

layer1 = od.Layer(10)
layer1.initialize_neuron_connections(5)
layer1.randomize_layer_weights()
layer1.randomize_bias_weights()
for i in range(len(layer1.neurons)):
    layer1.neurons[i].initialize_value(random.random())
network.add_layer(layer1)
network.generate_next_layer(layer1)
network.layer_list[1].initialize_neuron_connections(10)
network.layer_list[1].randomize_layer_weights()
network.layer_list[1].randomize_bias_weights()
print(network.layer_list[1].get_weighted_inputs())
network.generate_next_layer(network.layer_list[1])
network.backprop([0,1,0,0,0,0,0,0,0,0])
print(network)

