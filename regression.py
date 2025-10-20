import abc
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
        return 0.5 * mse

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        preds = X.dot(w)
        err = preds - y
        grad = X.T.dot(err) / float(m)
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
        # для оценки loss используем полный набор
        preds = X.dot(w)
        mse = np.mean((preds - y) ** 2)
        return 0.5 * mse

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


class LinearRegression:
    """
    Общий класс линейной регрессии, принимает реализацию GDMethod.
    """

    def __init__(self, method: BaseLoss, lr: float = 0.01, epochs: int = 1000):
        self.method = method
        self.lr = float(lr)
        self.epochs = int(epochs)

        self.theta: np.ndarray | None = None
        self.history: list[float] = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        m = X.shape[0]
        return np.hstack([np.ones((m, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_b = self._add_bias(X)
        n_params = X_b.shape[1]
        theta = np.zeros(n_params, dtype=float)

        self.history = []
        for epoch in range(self.epochs):
            grad = self.method.calc_grad(X_b, y, theta)
            if grad.shape != theta.shape:
                raise ValueError("gradient shape mismatch")
            theta = theta - self.lr * grad

            loss = float(self.method.calc_loss(X_b, y, theta))
            self.history.append(loss)

        self.theta = theta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise RuntimeError("model is not fitted")
        X_b = self._add_bias(X)
        return X_b.dot(self.theta)

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


if __name__ == "__main__":
    # Пример использования для BatchGD и SGD
    rng = np.random.default_rng(0)
    X = 2 * rng.random((200, 2))
    true_coef = np.array([3.0, -1.5])
    true_intercept = 2.0
    y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    # batch
    batch_method = MSELoss()
    model_batch = LinearRegression(method=batch_method, lr=0.1, epochs=300)
    model_batch.fit(X, y)
    preds_batch = model_batch.predict(X)
    print("batch theta:", model_batch.theta)
    print("batch R2:", model_batch.r2_score(X, y))

    # sgd
    sgd_method = SGD(rng=np.random.default_rng(1))
    model_sgd = LinearRegression(method=sgd_method, lr=0.01, epochs=5000)
    model_sgd.fit(X, y)
    preds_sgd = model_sgd.predict(X)
    print("sgd theta:", model_sgd.theta)
    print("sgd R2:", model_sgd.r2_score(X, y))

    np.random.seed(42) # для воспроизводимости результатов
    X = 2 * np.random.rand(100, 1)  # 100 примеров, 1 признак
    y = 4 + 3 * X + np.random.randn(100, 1) # Истинные веса: w=3, b=4

    model_batch = LinearRegression(method=batch_method, lr=0.1, epochs=300)
    model_batch.fit(X, y)
    print("batch 1 theta:", model_batch.theta)
    print("batch 1 R2:", model_batch.r2_score(X, y))

    sgd_method = SGD(rng=np.random.default_rng(1))
    model_sgd = LinearRegression(method=sgd_method, lr=0.01, epochs=5000)
    model_sgd.fit(X, y)
    print("sgd 1 theta:", model_sgd.theta)
    print("sgd 1 R2:", model_sgd.r2_score(X, y))