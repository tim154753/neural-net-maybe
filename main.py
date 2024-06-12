import mathfuncs as mf
import objectdefs as od
import random

print("New execution\n")

layer1 = od.Layer(5)
layer1.initialize_neuron_connections(4)
layer1.randomize_layer_weights()
for i in range(len(layer1.neurons)):
    layer1.neurons[i].initialize_value(random.random())
print(layer1)

layer2 = mf.generate_next_layer(layer1)
layer2.initialize_neuron_connections(10)
layer2.randomize_layer_weights()
print(layer2)

layer3 = mf.generate_next_layer(layer2)
print(layer3)

error = mf.proto_cost_function(9, layer3)
print(error)


