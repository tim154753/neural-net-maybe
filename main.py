import mathfuncs as mf
import objectdefs as od
import random

print("New execution\n")

layer1 = od.Layer(10)
layer1.initialize_neuron_connections(5)
layer1.randomize_layer_weights()
layer1.randomize_bias_weights()
for i in range(len(layer1.neurons)):
    layer1.neurons[i].initialize_value(random.random())
print(layer1)

layer2 = mf.generate_next_layer(layer1)
layer2.initialize_neuron_connections(10)
layer2.randomize_layer_weights()
layer2.randomize_bias_weights()
print(layer2)

layer3 = mf.generate_next_layer(layer2)
print(layer3)

error = mf.proto_cost_function(1, layer3, "avg")
print(error)





