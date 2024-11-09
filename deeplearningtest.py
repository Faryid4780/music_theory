import numpy as np
from deeplearning import Networks as Network
from deeplearning import SGD
from deeplearning import sigmoid
from deeplearning import mse_loss
import random

sigmoid12 = lambda x:12*sigmoid(x)

# 生成输入数据，每个样本有88个特征
num_samples = 20
input_dim = 88
output_dim = 17

network = Network(3, [88,12,12,17])
sgd = SGD(0.01)

# 生成随机输入数据
input_data = np.random.rand(num_samples, input_dim)

# 生成随机输出数据
output_data = np.random.randint(0, 2, size=(num_samples, output_dim))

# 输出数据在0到1之间，你可能需要进行一些后处理，确保它们在合适的范围内

# 打印数据集的形状
print("Input data shape:", input_data.shape)
print("Output data shape:", output_data.shape)

print(input_data)
print(output_data)

times = 1000
for i in range(times):
    print(i)
    for j,input_ in enumerate(input_data):
        d = network.gradients(mse_loss, input_, output_data[j], sigmoid12)
        sgd.step(network, d)

    target_index = random.randint(0,num_samples-1)
    to_input = input_data[target_index]
    to_output = output_data[target_index]
    print("Cost:",mse_loss(network.put(input_data,sigmoid12),to_output))

target_index = random.randint(0,num_samples-1)
to_input = input_data[target_index]
to_output = output_data[target_index]
print("Test:")
print("Target:",to_output)
print("Actual:",network.put(input_data,sigmoid12))



