import mathfuncs as mf
import objectdefs as od
import random

print("New execution\n")

'''network = od.Network(size = 3)

network.generate_first_layer(10)
network.layer_list[0].initialize_neuron_connections(5)
network.layer_list[0].randomize_layer_weights()
network.layer_list[0].randomize_bias_weights()
for i in range(len(network.layer_list[0].neurons)):
    network.layer_list[0].neurons[i].initialize_value(random.random())
network.generate_next_layer(network.layer_list[0])
network.layer_list[1].initialize_neuron_connections(3)
network.layer_list[1].randomize_layer_weights()
network.layer_list[1].randomize_bias_weights()
#print(network.layer_list[1].get_weighted_inputs())
network.generate_next_layer(network.layer_list[1])
print("\nBefore backprop\n")
print(network)
network.recursive_backprop([0,1,0,0,0,0,0,0,0,0])
print("After backprop\n")
print(network)'''

mf.read_image(0)
