import abc
from typing import List
import numpy as np


class BaseLoss(abc.ABC):
    @abc.abstractmethod
    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSELoss(BaseLoss):
    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        mse = np.mean((preds - y) ** 2)
        return mse

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        preds = X.dot(w)
        err = preds - y
        grad = 2 * X.T.dot(err) / float(m)
        return grad


class SGD(BaseLoss):
    """
    Стохастический градиентный спуск.
    Каждый вызов calc_grad выбирает один случайный пример и возвращает градиент по нему.
    Это даёт стохастические обновления, если в цикле fit вызывается calc_grad каждую итерацию.
    Опционально можно передать rng (numpy Generator) для детерминированности.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        mse = np.mean((preds - y) ** 2)
        return mse

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        if m == 0:
            return np.zeros_like(w)
        idx = int(self.rng.integers(0, m))
        Xi = X[idx:idx+1]        # shape (1, n+1)
        yi = y[idx:idx+1]         # shape (1,)
        pred = Xi.dot(w)      # shape (1,)
        err = pred - yi           # shape (1,)
        grad = Xi.T.dot(err)      # shape (n+1, 1)
        # grad для одного примера — усреднять на 1 не нужно, возвращаем одномерный вектор
        return grad.ravel()

def add_bias(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        m = X.shape[0]
        return np.hstack([np.ones((m, 1)), X])

class LinearRegression:
    """
    Общий класс линейной регрессии, принимает реализацию GDMethod.
    """

    def __init__(self, loss: BaseLoss, lr: float = 0.1, epochs: int = 1000):
        self.loss = loss
        self.lr = lr
        self.epochs = epochs

        self.w: np.ndarray | None = None
        self.history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        # X = np.asarray(X, dtype=float)
        # y = np.asarray(y, dtype=float).reshape(-1)
        # if X.shape[0] != y.shape[0]:
        #     raise ValueError("X and y must have same number of rows")

        # X_b = add_bias(X)
        # n_params = X_b.shape[1]
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        # Добавляем столбец из единиц для константного признака
        X = add_bias(X)

        w = np.zeros(X.shape[1], dtype=float)

        self.history = []
        for _ in range(self.epochs):
            grad = self.loss.calc_grad(X, y, w)
            if grad.shape != w.shape:
                raise ValueError("gradient shape mismatch")
            w = w - self.lr * grad

            loss = float(self.loss.calc_loss(X, y, w))
            self.history.append(loss)

        self.w = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("model is not fitted")
        X_b = add_bias(X)
        return X_b.dot(self.w)

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def gradient_descent(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    loss: BaseLoss,
    lr: float,
    n_iterations: int = 100000,
) -> List[np.ndarray]:
    """
    Функция градиентного спуска
    :param w_init: np.ndarray размера (n_feratures,) -- начальное значение вектора весов
    :param X: np.ndarray размера (n_objects, n_features) -- матрица объекты-признаки
    :param y: np.ndarray размера (n_objects,) -- вектор правильных ответов
    :param loss: Объект подкласса BaseLoss, который умеет считать градиенты при помощи loss.calc_grad(X, y, w)
    :param lr: float -- параметр величины шага, на который нужно домножать градиент
    :param n_iterations: int -- сколько итераций делать
    :return: Список из n_iterations объектов np.ndarray размера (n_features,) -- история весов на каждом шаге
    """

    history = []
    w = w_init
    for _ in range(n_iterations):
        grad = loss.calc_grad(X, y, w)
        w = w - lr * grad
        history.append(w)

    return history

def stochastic_gradient_descent(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    loss: BaseLoss,
    lr: float,
    batch_size: int,
    n_iterations: int = 1000,
) -> List[np.ndarray]:
    """
    Функция градиентного спуска
    :param w_init: np.ndarray размера (n_feratures,) -- начальное значение вектора весов
    :param X: np.ndarray размера (n_objects, n_features) -- матрица объекты-признаки
    :param y: np.ndarray размера (n_objects,) -- вектор правильных ответов
    :param loss: Объект подкласса BaseLoss, который умеет считать градиенты при помощи loss.calc_grad(X, y, w)
    :param lr: float -- параметр величины шага, на который нужно домножать градиент
    :param batch_size: int -- размер подвыборки, которую нужно семплировать на каждом шаге
    :param n_iterations: int -- сколько итераций делать
    :return: Список из n_iterations объектов np.ndarray размера (n_features,) -- история весов на каждом шаге
    """
    history = []
    w = w_init
    for _ in range(n_iterations):
        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        grad = loss.calc_grad(batch_X, batch_y, w)
        w = w - lr * grad
        history.append(w)

    return history

def stochastic_gradient_descent_step(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    loss: BaseLoss,
    lr: float,
    batch_size: int,
    p: float,
    n_iterations: int = 1000,
) -> List[np.ndarray]:
    """
    Функция градиентного спуска
    :param w_init: np.ndarray размера (n_feratures,) -- начальное значение вектора весов
    :param X: np.ndarray размера (n_objects, n_features) -- матрица объекты-признаки
    :param y: np.ndarray размера (n_objects,) -- вектор правильных ответов
    :param loss: Объект подкласса BaseLoss, который умеет считать градиенты при помощи loss.calc_grad(X, y, w)
    :param lr: float -- параметр величины шага, на который нужно домножать градиент
    :param batch_size: int -- размер подвыборки, которую нужно семплировать на каждом шаге
    :param p: float -- значение степени в формуле затухания длины шага
    :param n_iterations: int -- сколько итераций делать
    :return: Список из n_iterations объектов np.ndarray размера (n_features,) -- история весов на каждом шаге
    """
    s0 = 1000
    history = []
    w = w_init
    for i in range(n_iterations):
        batch_indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        grad = loss.calc_grad(batch_X, batch_y, w)
        step = lr * (s0 / (s0 + i)) ** p
        w = w - step * grad
        history.append(w)

    return history

if __name__ == "__main__":
    # # Пример использования для BatchGD и SGD
    # rng = np.random.default_rng(0)
    # X = 2 * rng.random((200, 2))
    # true_coef = np.array([3.0, -1.5])
    # true_intercept = 2.0
    # y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    # # batch
    batch_method = MSELoss()
    # model_batch = LinearRegression(method=batch_method, lr=0.1, epochs=300)
    # model_batch.fit(X, y)
    # preds_batch = model_batch.predict(X)
    # print("batch theta:", model_batch.theta)
    # print("batch R2:", model_batch.r2_score(X, y))

    # # sgd
    sgd_method = SGD(rng=np.random.default_rng(1))
    # model_sgd = LinearRegression(method=sgd_method, lr=0.01, epochs=5000)
    # model_sgd.fit(X, y)
    # preds_sgd = model_sgd.predict(X)
    # print("sgd theta:", model_sgd.theta)
    # print("sgd R2:", model_sgd.r2_score(X, y))

    np.random.seed(42) # для воспроизводимости результатов
    X = 2 * np.random.rand(100, 1)  # 100 примеров, 1 признак
    y = 4 + 3 * X + np.random.randn(100, 1) # Истинные веса: w=3, b=4

    model_batch = LinearRegression(loss=batch_method, lr=0.1, epochs=300)
    model_batch.fit(X, y)
    print("batch 1 theta:", model_batch.w)
    print("batch 1 R2:", model_batch.r2_score(X, y))

    sgd_method = SGD(rng=np.random.default_rng(1))
    model_sgd = LinearRegression(loss=sgd_method, lr=0.01, epochs=5000)
    model_sgd.fit(X, y)
    print("sgd 1 theta:", model_sgd.w)
    print("sgd 1 R2:", model_sgd.r2_score(X, y))

    # Создадим объект лосса
    loss = MSELoss()

    # Создадим какой-то датасет
    X = np.arange(200).reshape(20, 10)
    y = np.arange(20)

    # Создадим какой-то вектор весов
    w = np.arange(10)

    # Выведем значение лосса и градиента на этом датасете с этим вектором весов
    print(loss.calc_loss(X, y, w))
    print(loss.calc_grad(X, y, w))

    # Проверка, что методы реализованы правильно
    assert loss.calc_loss(X, y, w) == 27410283.5, "Метод calc_loss реализован неверно"
    assert np.allclose(
        loss.calc_grad(X, y, w),
        np.array(
            [
                1163180.0,
                1172281.0,
                1181382.0,
                1190483.0,
                1199584.0,
                1208685.0,
                1217786.0,
                1226887.0,
                1235988.0,
                1245089.0,
            ]
        ),
    ), "Метод calc_grad реализован неверно"
    print("Всё верно!")

    # Создаём датасет из двух переменных и реального вектора зависимости w_true

    np.random.seed(1337)

    n_features = 2
    n_objects = 300
    batch_size = 10
    num_steps = 43

    w_true = np.random.normal(size=(n_features,))

    X = np.random.uniform(-5, 5, (n_objects, n_features))
    X *= (np.arange(n_features) * 2 + 1)[np.newaxis, :]
    y = X.dot(w_true) + np.random.normal(0, 1, (n_objects))
    w_init = np.random.uniform(-2, 2, (n_features))

    print(X.shape)
    print(y.shape)

    loss = MSELoss()
    w_list = gradient_descent(w_init, X, y, loss, 0.01, 100)
    print(loss.calc_loss(X, y, w_list[0]))
    print(loss.calc_loss(X, y, w_list[-1]))

    w_list = stochastic_gradient_descent(w_init, X, y, loss, 0.01, 100, 1000)
    print(loss.calc_loss(X, y, w_list[0]))
    print(loss.calc_loss(X, y, w_list[-1]))

    w_list = stochastic_gradient_descent_step(w_init, X, y, loss, 0.01, 100, 0.6, 1000)
    print(loss.calc_loss(X, y, w_list[0]))
    print(loss.calc_loss(X, y, w_list[-1]))
