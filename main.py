import mathfuncs as mf
import objectdefs as od


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
network.add_layer(network.generate_next_layer(network.layer_list[1]))
labels = []



batch_size = 10
for i in range(batch_size):
    labels.append(mf.read_label(i))
    #generates initial list of 10 labels

for i in range(10):
    for j in range(batch_size*i, batch_size*i+batch_size):
        labels[j % batch_size] = mf.read_label(j)
    print(f"Labels {batch_size * i} through {batch_size * i + batch_size}\n")
    network.batch_gradient_descent(labels, batch_size*i)

avg_cost = 0
count = 0

print(network.layer_list[2])

for i in range(1000):
    test_label = mf.read_label(i+10000)
    mf.read_image(network.layer_list[0], i+10000)
    network.regenerate_network()
    if mf.is_correct(mf.read_label(i+10000, "num"), network.layer_list[2]):
        count += 1
    avg_cost += 1/1000 * mf.proto_cost_function(test_label, network.layer_list[2], "avg")

print(f"Average cost over 500 samples is: {avg_cost}")
print(f"Classified {count} images correctly out of 100 -> {count/10}% accuracy")

