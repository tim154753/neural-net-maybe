import mathfuncs as mf
import objectdefs as od
import numpy as np
import matplotlib.pyplot as plt
import random

print("New execution\n")

network = od.Network(size = 3, learning_rate=0.01)

network.generate_first_layer(784)
network.layer_list[0].initialize_neuron_connections(100)
network.layer_list[0].randomize_layer_weights()
network.layer_list[0].randomize_bias_weights()
network.add_layer(network.generate_next_layer(network.layer_list[0]))
network.layer_list[1].initialize_neuron_connections(10)
network.layer_list[1].randomize_layer_weights()
network.layer_list[1].randomize_bias_weights()
labels = []





#for i in range(10000):
    #labels.append(mf.read_label(i))
#print(labels)

#for i in range(len(network.layer_list[0].neurons)):
    #network.layer_list[0].neurons[i].initialize_value(random.random())
#print(network.layer_list[1].get_weighted_inputs())
network.add_layer(network.generate_next_layer(network.layer_list[1]))

#network.batch_gradient_descent(labels)

#print(network.layer_list[0].get_neuron_weights())
#print("\nBefore backprop\n")
#print(f"\n\n{len(network.layer_list[0].get_neuron_weights())} \n {len(network.layer_list[0].get_neuron_weights()[0])}")
#network.recursive_backprop(mf.read_label(0))


#print("After backprop\n")
#print(f"\n{network.layer_list[0]}\n")

#network.batch_gradient_descent(labels)

#mf.to_file(str(network), "C:/Users/timma/Desktop/newtest.txt")

#print(f"\n{network.layer_list[0]}\n")
#print(f"\n{network.layer_list[0].weight_gradient}\n")

batch_size = 0
for i in range(batch_size):
    labels.append(mf.read_label(i))
    #generates initial list of 10 labels

for i in range(0):
    for j in range(batch_size*i, batch_size*i+batch_size):
        labels[j % batch_size] = mf.read_label(j)
    print(f"Labels {batch_size * i} through {batch_size * i + batch_size}\n")
    network.batch_gradient_descent(labels, batch_size*i)

avg_cost = 0
count = 0
for i in range(500):
    test_label = mf.read_label(i+10000)
    mf.read_image(network.layer_list[0], i+10000)
    network.regenerate_network()
    if mf.is_correct(mf.read_label(i+10000, "num"), network.layer_list[2]):
        count += 1
    avg_cost += 1/500 * mf.proto_cost_function(test_label, network.layer_list[2], "avg")

print(f"Average cost over 500 samples is: {avg_cost}")
print(f"Classified {count} images correctly out of 500 -> {count/5}% accuracy")
'''image_number = 15
mf.read_image(network.layer_list[0], image_number)
network.regenerate_network()

print(f"\n{network.layer_list[2]}\n")

print(mf.read_label(image_number))

result = f"After training, this is output with image #{image_number}, with correct array {str(mf.read_label(image_number))}:\n\n {str(network.layer_list[2])} \n\n"

mf.to_file(result, "C:/Users/timma/Desktop/newtest.txt")


'''

'''mf.read_image(network.layer_list[0], 5000)
network.recursive_backprop(mf.read_label(5000))
for layer in network.layer_list:
    layer.update_layer_weights()
    layer.update_bias_weights()
print(network.layer_list[0])'''

'''mf.read_image(network.layer_list[0], 2000)
print(mf.read_label(2000))
network.regenerate_network()
print(network.layer_list[0])
print(network.layer_list[2])'''
