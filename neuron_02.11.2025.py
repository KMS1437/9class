import numpy as np


def activation(x, threshold):
    return 1 if x >= threshold else 0


data = np.array([
    [155, 45, 18], [195, 95, 45], [145, 42, 16], [205, 110, 50], [135, 35, 14]
])

mean_value = np.mean(data)
threshold = mean_value - (0.05 * mean_value)


def predict(inputs, weights, bias):
    y = np.dot(inputs, weights) + bias
    return activation(y, threshold)


a, b, c = map(int, input("Введите рост вес и возраст через пробел: ").split())
person_data = np.array([a, b, c])

print(f"Среднее значение данных: {mean_value:.2f}")
print(f"Рассчитанный порог: {threshold:.2f}")
