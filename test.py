import numpy as np
# import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    Класс линейной регрессии, обучаемый с помощью градиентного спуска.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Инициализация модели.

        Args:
            learning_rate (float): Скорость обучения (alpha).
            n_iterations (int): Количество итераций градиентного спуска.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Веса (w)
        self.bias = None     # Смещение (b)
        self.cost_history = [] # История значений функции стоимости для отладки

    def fit(self, X, y):
        """
        Обучение модели на данных X и y.

        Args:
            X (np.ndarray): Матрица признаков (m_samples, n_features).
            y (np.ndarray): Вектор целевых значений (m_samples, 1).
        """
        # 1. Инициализация весов и смещения
        m, n = X.shape  # m - количество примеров, n - количество признаков
        self.weights = np.zeros((n, 1)) # Инициализируем веса нулями
        self.bias = 0                      # Инициализируем смещение нулем

        # Преобразуем y в вектор-столбец, если он им не является
        y = y.reshape(m, 1)

        # 2. Градиентный спуск
        for i in range(self.n_iterations):
            # a. Вычисление предсказаний (гипотеза)
            # y_pred = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # b. Вычисление градиентов
            # Формула градиента для MSE:
            # dw = (2/m) * X.T * (y_pred - y)
            # db = (2/m) * sum(y_pred - y)
            # Множитель 2 можно "спрятать" в learning_rate, что обычно и делается.
            error = y_predicted - y
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)

            # c. Обновление весов и смещения
            # w = w - alpha * dw
            # b = b - alpha * db
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # d. Сохранение значения функции стоимости для построения графика
            cost = np.mean((error)**2)
            self.cost_history.append(cost)

            # (Опционально) Вывод прогресса
            if (i+1) % 100 == 0:
                print(f"Итерация {i+1}/{self.n_iterations}, Ошибка (MSE): {cost:.4f}")

    def predict(self, X):
        """
        Делает предсказания для новых данных X.

        Args:
            X (np.ndarray): Матрица признаков для предсказания.

        Returns:
            np.ndarray: Вектор предсказанных значений.
        """
        return np.dot(X, self.weights) + self.bias

# --- 1. Генерация данных ---
# Создадим данные с линейной зависимостью y = 4 + 3x + шум
np.random.seed(42) # для воспроизводимости результатов
X = 2 * np.random.rand(100, 1)  # 100 примеров, 1 признак
y = 4 + 3 * X + np.random.randn(100, 1) # Истинные веса: w=3, b=4

# --- 2. Создание и обучение модели ---
# Создаем экземпляр класса с гиперпараметрами
model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)

# Обучаем модель на наших данных
print("Начало обучения...")
model.fit(X, y)
print("Обучение завершено.")

# --- 3. Вывод результатов ---
print("\nНайденные параметры:")
print(f"Вес (w): {model.weights[0][0]:.4f}") # Ожидаем ~3.0
print(f"Смещение (b): {model.bias:.4f}")      # Ожидаем ~4.0

# --- 4. Визуализация результатов ---
# Создаем фигуру с двумя графиками
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # График 1: Исходные данные и линия регрессии
# ax1.scatter(X, y, alpha=0.7, label='Исходные данные')
# # Предсказываем значения для построения линии
# X_new = np.array([[0], [2]]) # Точки для построения линии
# y_predict = model.predict(X_new)
# ax1.plot(X_new, y_predict, "r-", linewidth=2, label='Линия регрессии')
# ax1.set_xlabel("Признак (X)")
# ax1.set_ylabel("Целевая переменная (y)")
# ax1.set_title("Линейная регрессия")
# ax1.legend()
# ax1.grid(True)

# График 2: История функции стоимости
# ax2.plot(range(model.n_iterations), model.cost_history)
# ax2.set_xlabel("Итерация")
# ax2.set_ylabel("Ошибка (MSE)")
# ax2.set_title("График функции стоимости")
# ax2.grid(True)

# plt.show()
