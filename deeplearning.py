import numpy as np
import math

# 初始化参数
input_size = 3
hidden_size = 4
output_size = 2
# 定义激活函数（这里使用 sigmoid 函数）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NetworkLayer:
    def __init__(self,input_size:int,output_size:int):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(1, output_size)
    def acti_function(self,input_,method=None):
        """
        :param input_: 输入
        :param method: 自定义激活函数，类型是一个function，None默认为sigmoid
        :return: 经过激活函数后的值
        """
        if method==None:
            return sigmoid(input_)
        return method(input_)
    def put(self, input_:np.array, method=None):
        """
        :param input_: 输入数据
        :return: 输出数据
        """
        h = np.dot(input_, self.weights)+self.biases
        return self.acti_function(h,method)


class Networks:
    def __init__(self, layers:int, count:list):
        assert layers+1==len(count)
        self.layers = []
        for i in range(layers):
            self.layers.append(NetworkLayer(count[i],count[i+1]))
            print(count[i],count[i+1])

    def put(self, input_: np.array, method=None):
        """
        :param input_: 输入数据
        :return: 输出数据
        """
        h = input_.copy()
        for layer in self.layers:
            h = layer.put(h,method)
        return h

    def gradients(self, losing_method, input_array:np.array,correct_array:np.array, method=None):
        losing_points = losing_method(self.put(input_array, method), correct_array)

        derivative = []
        delta = 0.0001
        n = 0
        for i in self.layers:
            derivative_to_layer = np.zeros((i.input_size,i.output_size))
            for ix in range(i.weights.shape[0]):
                for iy in range(i.weights.shape[1]):
                    i.weights[ix][iy] += delta
                    losing_points_now = losing_method(self.put(input_array, method), correct_array)
                    derivative_to_layer[ix][iy] = losing_points_now-losing_points
                    i.weights[ix][iy] -= delta
            n+=1
            derivative.append(derivative_to_layer)

        return derivative


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, network:Networks, gradients:list):
        for i,layer in enumerate(network.layers):
            layer.weights -= gradients[i] * self.learning_rate


# 定义损失函数（这里使用均方误差）
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


if __name__ == '__main__':
    # 定义样本数据
    X = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6]])
    y_true = np.array([[0.7, 0.8],
                       [0.2, 0.3]])

    # 前向传播
    network = Networks(2,[input_size,hidden_size,output_size])
    y_pred = network.put(X)

    # 计算损失
    loss = mse_loss(y_pred, y_true)
    for x in range(100):
        print(network.gradients(mse_loss, X, y_true, sigmoid))


        print("预测值：")
        print(y_pred)
        print("损失：", loss)
