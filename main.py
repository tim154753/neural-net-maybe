import mathfuncs as mf
import objectdefs as od
import random

print("New execution\n")

network = od.Network(size = 3)

network.generate_first_layer(10)
network.layer_list[0].initialize_neuron_connections(5)
network.layer_list[0].randomize_layer_weights()
network.layer_list[0].randomize_bias_weights()
network.add_layer(network.generate_next_layer(network.layer_list[0]))
network.layer_list[1].initialize_neuron_connections(3)
network.layer_list[1].randomize_layer_weights()
network.layer_list[1].randomize_bias_weights()
labels = []
for i in range(100):
    labels.append(mf.read_label(i))

#for i in range(len(network.layer_list[0].neurons)):
    #network.layer_list[0].neurons[i].initialize_value(random.random())
#print(network.layer_list[1].get_weighted_inputs())
network.add_layer(network.generate_next_layer(network.layer_list[1]))

network.batch_gradient_descent(labels)


#print("\nBefore backprop\n")
print(network)
#network.recursive_backprop(mf.read_label(0))
#print("After backprop\n")
#print(network)
