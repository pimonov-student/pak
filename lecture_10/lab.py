import numpy as np


# Создадим класс для нейрона
class Neuron:
    # Инициилизируем
    def __init__(self, input_number, output_number):
        self.weights = np.random.uniform(size=(input_number, output_number))
        self.bias = np.random.uniform(1, output_number)
        self.output = 0
    
    # Функция активации (сигмоида, очевидно)
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    
    # Прямой проход по нейрону
    def forward(self, input):
        neuron_activation = np.dot(input, self.weights) + self.bias
        self.output = self.sigmoid(neuron_activation)
        return self.output
    
    # Обратный проход по нейрону
    def backward(self, input, loss):
        self.weights += np.dot(input, loss)
        self.bias += np.sum(loss, axis=0, keepdims=True)


# Создадим класс модели
class Model:
    # Инициилизируем
    def __init__(self, input_number, hidden_number, output_number, expected_output):
        self.hidden_layer = Neuron(input_number, hidden_number)
        self.output_layer = Neuron(hidden_number, output_number)
        self.expected_output = expected_output
        self.predicted_output = 0
    
    # Производная от сигмоиды, потребуется при обратном проходе
    def sigmoid_derivative(self, x):
        return (x * (1 - x))

    # Прямой проход
    def forward(self, input):
        hidden_output = self.hidden_layer.forward(input)
        self.predicted_output = self.output_layer.forward(hidden_output)
        return self.predicted_output
    
    # Обратный проход
    def backward(self, input):
        error = self.expected_output - self.predicted_output
        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)

        error_hidden = d_predicted_output.dot(self.output_layer.weights.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_layer.output)

        self.output_layer.backward(self.hidden_layer.output.T, d_predicted_output)
        self.hidden_layer.backward(input.T, d_hidden)
    
    # Тренируем
    def train(self, input, count):
        for _ in range(count):
            self.forward(input)
            self.backward(input)
    
    # Предсказываем
    def predict(self, input):
        return self.forward(input)
    
    # Считаем точность
    def accuracy(self, predicted, expected):
        return (1 - np.abs(expected - predicted))

# Проверим

# Зададим входные данные и ожидаемые выходы
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Зададим параметры нашей модели
input_number = 2
hidden_number = 2
output_number = 1

# Создадим модель и натренируем ее
model = Model(input_number, hidden_number, output_number, expected_output)
model.train(input, 10000)

# Проверим
predicted_output = model.predict([[1, 0], [1, 1]])
print(model.predicted_output)
print(model.accuracy(model.predicted_output, [[1], [0]]))
