import numpy as np

# Задание 1

# Напишем класс для слоев, реализуем инициализацию слоя и прямой проход
class FullyConnectedLayer:
    def __init__(self, input_data_number, neuron_number):
        self.weights = np.random.normal(size=(input_data_number, neuron_number))
        self.bias = np.random.normal(size=neuron_number)
    
    def forward(self, x, activate_function_type):
        output = np.dot(x, self.weights) + self.bias
        
        if (activate_function_type == 'relu'):
            return np.maximum(0, output)
        elif (activate_function_type == 'tanh'):
            return np.tanh(output)
        elif (activate_function_type == 'softmax'):
            return (np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output))))

# Напишем теперь класс полносвязной модели, состоящей из трех слоев
class FullyConnectedModel:
    def __init__(self,
                 input_data_number,
                 layer_one_neuron_number, 
                 layer_two_neuron_number, 
                 output_layer_neuron_number):
        self.layer_one = FullyConnectedLayer(input_data_number, layer_one_neuron_number)
        self.layer_two = FullyConnectedLayer(layer_one_neuron_number, layer_two_neuron_number)
        self.output_layer = FullyConnectedLayer(layer_two_neuron_number, output_layer_neuron_number)
    
    def forward(self, x):
        layer_one_output = self.layer_one.forward(x, 'relu')
        layer_two_output = self.layer_two.forward(layer_one_output, 'tanh')
        output = self.output_layer.forward(layer_two_output, 'softmax')
        return output

# Задание 2

# Напишем класс сверточного слоя
class ConvolutionLayer:
    def __init__(self, filters):
        self.filters = filters
    
    def convolve(self, matrix):
        filter_number, filter_x, filter_y = self.filters.shape
        matrix_channels, matrix_x, matrix_y = matrix.shape
        res_matrix_x = matrix_x - 1
        res_matrix_y = matrix_y - 1

        result = np.zeros(filter_number * res_matrix_x * res_matrix_y).reshape(filter_number, res_matrix_x, res_matrix_y)

        for f in range(filter_number):
            for c in range(matrix_channels):
                for x in range(res_matrix_x):
                    for y in range(res_matrix_y):
                        result[f][x][y] = np.sum(matrix[c][x : x + filter_x, y : y + filter_y] * self.filters[f])
        
        return result
    
    def max_pool(self, matrix):
        matrix_channels, matrix_x, matrix_y = matrix.shape
        res_matrix_x = matrix_x // 2
        res_matrix_y = matrix_y // 2

        result = np.zeros(matrix_channels * res_matrix_x * res_matrix_y).reshape(matrix_channels, res_matrix_x, res_matrix_y)

        for c in range(matrix_channels):
            for x in range(res_matrix_x):
                for y in range(res_matrix_y):
                    result[c][x][y] = np.max(matrix[c][2 * x : 2 * x + 1, 2 * y : 2 * y + 1])
        
        return result

# Напишем класс сверточной модели
class ConvolutionModel:
    def __init__(self,
                 layer_one_filters,
                 layer_two_filters):
        self.layer_one = ConvolutionLayer(layer_one_filters)
        self.layer_two = ConvolutionLayer(layer_two_filters)
    
    def forward(self, matrix):
        matrix = self.layer_one.convolve(matrix)
        matrix = self.layer_one.max_pool(matrix)
        matrix = self.layer_two.convolve(matrix)
        matrix = self.layer_two.max_pool(matrix)

        return matrix

# Задание 3

# Свяжем обе модели
class Model:
    def __init__(self,
                 layer_one_filters, layer_two_filters,
                 input_data_number,
                 layer_one_neuron_number, 
                 layer_two_neuron_number, 
                 output_layer_neuron_number):
        self.fc = FullyConnectedModel(input_data_number, layer_one_neuron_number, layer_two_neuron_number, output_layer_neuron_number)
        self.conv = ConvolutionModel(layer_one_filters, layer_two_filters)
    
    def forward(self, matrix):
        result = self.conv.forward(matrix)
        result = self.fc.forward(result.reshape(-1))

        return result

# Проверим

# Создадим случайные входные данные
input = np.random.normal(size=(3, 19, 19))
# Теперь создадим случайные фильтры для свертки
layer_one_filters = np.random.normal(size=(8, 2, 2))
layer_two_filters = np.random.normal(size=(16, 2, 2))
# Зададим количество входов и нейронов для FC-модели
input_data_number = 256
layer_one_neuron_number = 64
layer_two_neuron_number = 16
output_layer_neuron_number = 4

# Создадим модель
model = Model(layer_one_filters, layer_two_filters,
              input_data_number, layer_one_neuron_number, layer_two_neuron_number, output_layer_neuron_number)

# Запустим прямой проход и получим вывод
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
result = model.forward(input)
print(result)
