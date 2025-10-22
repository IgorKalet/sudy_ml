IgorKalet: хапрограммировати линейную регрессию методом градиентного спуска с использованием numpy

GitHub Copilot: Copilot took too long to get ready. Please ensure you are signed in to GitHub and that the extension `GitHub.copilot-chat` is installed and enabled.

IgorKalet: реализуй линейную регрессию методои градиентного спуска с numpy

GitHub Copilot: Ниже — компактная реализация линейной регрессии методом (пакетного) градиентного спуска на numpy + пример использования.

````python
# python
import numpy as np

class LinearRegressionGD:
    """
    Линейная регрессия через градиентный спуск (batch).
    fit(X, y, lr, n_iter, tol, verbose)
    После fit доступны атрибуты coef_ (вектор коэффициентов) и intercept_ (смещение).
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.history_ = None  # loss history

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y, lr=0.01, n_iter=1000, tol=1e-6, verbose=False):
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        m, n = X.shape

        # Добавим столбец единиц для смещения и объединяем параметры в theta
        X_b = np.hstack([np.ones((m, 1)), X])  # shape (m, n+1)
        theta = np.zeros(n + 1, dtype=float)   # [intercept, coef...]

        self.history_ = []
        prev_loss = None

        for i in range(n_iter):
            preds = X_b.dot(theta)
            error = preds - y

            # MSE и градиент
            loss = (error**2).mean() / 2.0
            grad = (X_b.T.dot(error)) / m

            theta -= lr * grad

            self.history_.append(loss)
            if verbose and (i % max(1, n_iter // 10) == 0):
                print(f"iter {i:5d} loss={loss:.6f}")

            if prev_loss is not None and abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        return self

    def predict(self, X):
        X = self._prepare_X(X)
        return self.intercept_ + X.dot(self.coef_)

    def score_r2(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

# Пример использования
if __name__ == "__main__":
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    true_intercept = 4.0
    true_coef = np.array([3.0])
    y = true_intercept + X.dot(true_coef)[:,0] + np.random.randn(100) * 0.5

    model = LinearRegressionGD()
    model.fit(X, y, lr=0.1, n_iter=2000, verbose=True)
    print("intercept:", model.intercept_)
    print("coef:", model.coef_)
    print("R2:", model.score_r2(X, y))
````

Если нужно добавить нормализацию признаков, стохастический / мини-батч градиентный спуск или вывод графика истории loss — скажите, добавлю.

IgorKalet: добавить нормализацию признаков, стохастический / мини-батч градиентный спуск или вывод графика истории loss

GitHub Copilot: Добавил новый Python-файл с реализацией линейной регрессии через градиентный спуск: поддерживается нормализация признаков, пакетный / стохастический / мини-батч градиентный спуск и вывод графика истории loss.

Файл: d:\test\git_new\roofline\linear_regression_gd.py

````python
import numpy as np


class LinearRegressionGD:
    """
    Линейная регрессия через градиентный спуск.
    Параметры:
      method: 'batch' | 'sgd' | 'minibatch'
      lr: learning rate
      epochs: количество проходов по данным (эпох)
      batch_size: размер мини-батча (при method='minibatch')
      normalize: стандартная нормализация (z-score) признаков
      shuffle: перемешивать данные каждую эпоху (для sgd/minibatch)
      record_per_step: если True — логируется loss для каждого шага (может быть много точек)
    Результат:
      coef_: numpy array коэффициентов (без сдвига)
      intercept_: смещение
      history_: список loss
    """
    def __init__(self, method="batch", lr=0.01, epochs=1000, batch_size=32,
                 normalize=True, shuffle=True, tol=1e-8, verbose=False, record_per_step=False):
        self.method = method
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.normalize = normalize
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.record_per_step = record_per_step

        self.coef_ = None
        self.intercept_ = None
        self.history_ = []

        # normalization params
        self.x_mean_ = None
        self.x_std_ = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _normalize(self, X, fit=False):
        if not self.normalize:
            return X
        if fit:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0.0] = 1.0
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        m, n = X.shape

        X = self._normalize(X, fit=True)

        # добавляем столбец единиц для смещения
        X_b = np.hstack([np.ones((m, 1)), X])  # shape (m, n+1)
        theta = np.zeros(n + 1, dtype=float)   # [intercept, coef...]

        self.history_ = []
        prev_loss = None

        rng = np.random.default_rng(42)

        for epoch in range(self.epochs):
            if self.method == "batch":
                preds = X_b.dot(theta)
                error = preds - y
                loss = 0.5 * np.mean(error ** 2)
                grad = X_b.T.dot(error) / m
                theta -= self.lr * grad
                # record one value per epoch (or more if record_per_step)
                self.history_.append(loss)

            elif self.method == "sgd":
                indices = np.arange(m)
                if self.shuffle:
                    rng.shuffle(indices)
                # optionally record loss per sample
                for i in indices:
                    xi = X_b[i:i+1]  # shape (1, n+1)
                    err = xi.dot(theta) - y[i]
                    grad = xi.T.flatten() * err  # shape (n+1,)
                    theta -= self.lr * grad
                    if self.record_per_step:
                        # compute full loss occasionally or per step (expensive)
                        preds = X_b.dot(theta)
                        loss = 0.5 * np.mean((preds - y) ** 2)
                        self.history_.append(loss)
                if not self.record_per_step:
                    preds = X_b.dot(theta)
                    loss = 0.5 * np.mean((preds - y) ** 2)
                    self.history_.append(loss)

            elif self.method == "minibatch":
                indices = np.arange(m)
                if self.shuffle:
                    rng.shuffle(indices)
                for start in range(0, m, self.batch_size):
                    batch_idx = indices[start:start + self.batch_size]
                    Xb = X_b[batch_idx]
                    yb = y[batch_idx]
                    preds = Xb.dot(theta)
                    err = preds - yb
                    grad = Xb.T.dot(err) / Xb.shape[0]
                    theta -= self.lr * grad
                    if self.record_per_step:
                        preds_full = X_b.dot(theta)
                        loss_full = 0.5 * np.mean((preds_full - y) ** 2)
                        self.history_.append(loss_full)
                if not self.record_per_step:
                    preds = X_b.dot(theta)
                    loss = 0.5 * np.mean((preds - y) ** 2)
                    self.history_.append(loss)
            else:
                raise ValueError("unknown method: " + str(self.method))

            # verbose / early stop
            loss = self.history_[-1]
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"epoch {epoch:4d} loss={loss:.6f}")
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print("converged, breaking")
                break
            prev_loss = loss

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        return self

    def predict(self, X):
        X = self._prepare_X(X)
        if self.normalize and (self.x_mean_ is not None):
            X = (X - self.x_mean_) / self.x_std_
        return self.intercept_ + X.dot(self.coef_)

    def score_r2(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def plot_loss(self, show=True, save_path=None):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib required for plot_loss, install with: pip install matplotlib") from e

        if not self.history_:
            raise RuntimeError("history is empty, fit model first")

        plt.figure()
        plt.plot(self.history_, marker=".", linewidth=0.8)
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss (0.5 * MSE)")
        plt.title(f"Loss history ({self.method})")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    # Пример использования
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    # синтетические данные
    X = 2 * rng.random((200, 2))  # два признака
    true_coef = np.array([3.0, -1.5])
    true_intercept = 2.0
    y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    # batch
    model_batch = LinearRegressionGD(method="batch", lr=0.1, epochs=200, normalize=True, verbose=True)
    model_batch.fit(X, y)
    print("batch intercept:", model_batch.intercept_, "coef:", model_batch.coef_, "R2:", model_batch.score_r2(X, y))

    # minibatch
    model_mb = LinearRegressionGD(method="minibatch", lr=0.05, epochs=200, batch_size=32,
                                  normalize=True, shuffle=True, verbose=True, record_per_step=False)
    model_mb.fit(X, y)
    print("minibatch intercept:", model_mb.intercept_, "coef:", model_mb.coef_, "R2:", model_mb.score_r2(X, y))
    model_mb.plot_loss(show=True)
````

Запуск (Windows):
- Убедитесь, что установлен numpy (+ matplotlib, если хотите график): pip install numpy matplotlib
- Выполните: python d:\test\git_new\roofline\linear_regression_gd.py

Если хотите — внесу изменения в существующий Python-скрипт в проекте или адаптирую поведение (адаптивный lr, регуляризация, масштабирование y и т.д.).

IgorKalet: добавь абстрактный класс для градиентного спуска, все реализации градиентного спуска должны наследоваться от данного абстрактного класса. Линейная регрессия должна содержать только вызовы данного абстрактного класса.

GitHub Copilot: Добавил абстрактный базовый класс для градиентного спуска и переработал линейную регрессию: все реализации GD наследуются от базового класса, а LinearRegressionGD теперь только вызывает реализацию градиентного спуска.

Файлы, которые добавлены/изменены:

1) Новый базовый класс и конкретные реализации (batch / sgd / minibatch)
````python
import abc
import numpy as np


class GradientDescentBase(abc.ABC):
    """
    Абстрактный базовый класс для методов градиентного спуска.
    Конкретные реализации должны реализовать _run_epoch, который вызывает
    переданную функцию grad_func для вычисления градиента по батчу.
    """

    def __init__(self, lr=0.01, epochs=1000, batch_size=32, shuffle=True,
                 tol=1e-8, verbose=False, record_per_step=False, rng_seed=42):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.record_per_step = bool(record_per_step)
        self.history_ = []
        self.rng = np.random.default_rng(rng_seed)

    def fit(self, X_b, y, theta_init, grad_func, loss_func=None):
        """
        Общая логика обучения.
        - X_b: матрица признаков с единичным столбцом для смещения (m, n+1)
        - y: целевая вектор (m,)
        - theta_init: начальные параметры (n+1,)
        - grad_func: функция (X_batch, y_batch, theta) -> grad (n+1,)
        - loss_func: опциональная функция (X_b, y, theta) -> loss (скаляр)
        Возвращает theta (numpy array).
        """
        m = X_b.shape[0]
        theta = theta_init.copy()
        self.history_.clear()
        prev_loss = None

        for epoch in range(self.epochs):
            # реализация эпохи делегируется конкретному методу
            self._run_epoch(X_b, y, theta, grad_func)

            # вычисление loss для контроля/ранней остановки
            if loss_func is not None:
                loss = float(loss_func(X_b, y, theta))
            else:
                # если loss_func не передан, попробуем оценить как 0.5 * MSE
                preds = X_b.dot(theta)
                loss = 0.5 * np.mean((preds - y) ** 2)

            self.history_.append(loss)
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"epoch {epoch:4d} loss={loss:.6f}")

            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print("converged, breaking")
                break
            prev_loss = loss

        return theta

    @abc.abstractmethod
    def _run_epoch(self, X_b, y, theta, grad_func):
        """
        Реализует поведение одной эпохи: должна изменять theta in-place используя grad_func.
        grad_func(X_batch, y_batch, theta) -> gradient
        """
        raise NotImplementedError


class BatchGD(GradientDescentBase):
    def _run_epoch(self, X_b, y, theta, grad_func):
        m = X_b.shape[0]
        grad = grad_func(X_b, y, theta)
        theta -= self.lr * (grad / float(m))


class SGD(GradientDescentBase):
    def _run_epoch(self, X_b, y, theta, grad_func):
        m = X_b.shape[0]
        indices = np.arange(m)
        if self.shuffle:
            self.rng.shuffle(indices)
        for i in indices:
            Xi = X_b[i:i+1]
            yi = y[i:i+1]
            grad = grad_func(Xi, yi, theta)
            # grad returned is for batch of size 1 -> no division by m here
            theta -= self.lr * grad
            if self.record_per_step:
                # можно оставить место для записи промежуточного loss (в fit)
                pass


class MiniBatchGD(GradientDescentBase):
    def _run_epoch(self, X_b, y, theta, grad_func):
        m = X_b.shape[0]
        indices = np.arange(m)
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, m, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            Xbatch = X_b[batch_idx]
            ybatch = y[batch_idx]
            grad = grad_func(Xbatch, ybatch, theta)
            theta -= self.lr * (grad / float(max(1, Xbatch.shape[0])))
````

2) Изменённый файл линейной регрессии: вызывает GradientDescent реализации и не содержит цикл градиентного спуска
````python
import numpy as np
from gradient_descent import BatchGD, SGD, MiniBatchGD


class LinearRegressionGD:
    """
    Линейная регрессия. Логика градиентного спуска делегируется классам из gradient_descent.py.
    Параметры и поведение аналогичны прежней реализации, но без собственного цикла обучения.
    """
    def __init__(self, method="batch", lr=0.01, epochs=1000, batch_size=32,
                 normalize=True, shuffle=True, tol=1e-8, verbose=False, record_per_step=False):
        self.method = method
        self.lr = lr
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.normalize = normalize
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.record_per_step = record_per_step

        self.coef_ = None
        self.intercept_ = None
        self.history_ = []

        # normalization params
        self.x_mean_ = None
        self.x_std_ = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _normalize(self, X, fit=False):
        if not self.normalize:
            return X
        if fit:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_ = X.std(axis=0)
            self.x_std_[self.x_std_ == 0.0] = 1.0
        return (X - self.x_mean_) / self.x_std_

    def _make_gd_solver(self):
        kwargs = dict(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                      shuffle=self.shuffle, tol=self.tol, verbose=self.verbose,
                      record_per_step=self.record_per_step)
        if self.method == "batch":
            return BatchGD(**kwargs)
        if self.method == "sgd":
            return SGD(**kwargs)
        if self.method == "minibatch":
            return MiniBatchGD(**kwargs)
        raise ValueError("unknown method: " + str(self.method))

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        m, n = X.shape

        X = self._normalize(X, fit=True)
        X_b = np.hstack([np.ones((m, 1)), X])  # (m, n+1)
        theta_init = np.zeros(n + 1, dtype=float)

        # определяем функции градиента и loss для линейной регрессии
        def grad_func(X_batch, y_batch, theta):
            preds = X_batch.dot(theta)
            err = preds - y_batch
            # gradient for batch (sum of errors * X) -> shape (n+1,)
            return X_batch.T.dot(err)

        def loss_func(X_all, y_all, theta):
            preds = X_all.dot(theta)
            return 0.5 * np.mean((preds - y_all) ** 2)

        solver = self._make_gd_solver()
        theta = solver.fit(X_b, y, theta_init, grad_func, loss_func)

        # сохранить историю и параметры
        self.history_ = solver.history_
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        return self

    def predict(self, X):
        X = self._prepare_X(X)
        if self.normalize and (self.x_mean_ is not None):
            X = (X - self.x_mean_) / self.x_std_
        return self.intercept_ + X.dot(self.coef_)

    def score_r2(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = self.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def plot_loss(self, show=True, save_path=None):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise RuntimeError("matplotlib required for plot_loss, install with: pip install matplotlib")

        if not self.history_:
            raise RuntimeError("history is empty, fit model first")

        plt.figure()
        plt.plot(self.history_, marker=".", linewidth=0.8)
        plt.xlabel("Step / Epoch")
        plt.ylabel("Loss (0.5 * MSE)")
        plt.title(f"Loss history ({self.method})")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    # Пример использования (как и раньше)
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    X = 2 * rng.random((200, 2))
    true_coef = np.array([3.0, -1.5])
    true_intercept = 2.0
    y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    model_batch = LinearRegressionGD(method="batch", lr=0.1, epochs=200, normalize=True, verbose=True)
    model_batch.fit(X, y)
    print("batch intercept:", model_batch.intercept_, "coef:", model_batch.coef_, "R2:", model_batch.score_r2(X, y))

    model_mb = LinearRegressionGD(method="minibatch", lr=0.05, epochs=200, batch_size=32,
                                  normalize=True, shuffle=True, verbose=True, record_per_step=False)
    model_mb.fit(X, y)
    print("minibatch intercept:", model_mb.intercept_, "coef:", model_mb.coef_, "R2:", model_mb.score_r2(X, y))
    model_mb.plot_loss(show=True)
````

Если нужно — внесу изменения в существующий файл (вместо создания новых) или добавлю типы/аннотации, unit-тесты, или переведу дизайн на чистую объектно-ориентированную модель где LinearRegression наследует от абстрактного класса.

IgorKalet: Запрограммируй общий класс линейной регрессии, который принимает конкретный метод градиентного спуска в качестве параметра. Пусть этот параметр является абстрактным классом с двумя методами calc_loss для вычисления лоса (потерь) и calc_grad для вычисления градиента.

GitHub Copilot: Добавил общий класс линейной регрессии и абстрактный интерфейс для метода градиентного спуска + реализацию пакетного (batch) GD. Код — в новом пакете ml.

````go
package ml

import (
	"fmt"
	"math"
)

// GDMethod — абстрактный интерфейс метода градиентного спуска.
// CalcLoss принимает матрицу признаков с колонкой единиц (X_b), вектор y и параметры theta,
// возвращает скалярный loss.
// CalcGrad принимает те же аргументы и возвращает градиент по theta (размер len(theta)).
// Контракт: CalcGrad должен возвращать усреднённый градиент (т.е. уже делить на m), чтобы
// шаг обновления был просто theta -= lr * grad.
type GDMethod interface {
	CalcLoss(Xb [][]float64, y []float64, theta []float64) float64
	CalcGrad(Xb [][]float64, y []float64, theta []float64) []float64
}

// LinearRegression — общий класс линейной регрессии, принимает конкретный метод GD.
type LinearRegression struct {
	Theta        []float64
	Method       GDMethod
	LearningRate float64
	Epochs       int
	History      []float64
}

// NewLinearRegression создаёт модель с переданным методом GD.
func NewLinearRegression(method GDMethod, lr float64, epochs int) *LinearRegression {
	return &LinearRegression{
		Method:       method,
		LearningRate: lr,
		Epochs:       epochs,
	}
}

// Fit обучает модель. Ожидается, что X — матрица m x n (без колонки единиц).
func (lrm *LinearRegression) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 {
		return fmt.Errorf("empty X")
	}
	m := len(X)
	n := len(X[0])

	if len(y) != m {
		return fmt.Errorf("len(y) must equal number of rows in X")
	}

	// X_b — добавляем колонку единиц слева
	Xb := addOnesColumn(X)

	// инициализация theta (n+1)
	theta := make([]float64, n+1)

	lrm.History = lrm.History[:0]

	for e := 0; e < lrm.Epochs; e++ {
		grad := lrm.Method.CalcGrad(Xb, y, theta) // должен быть усреднённый градиент
		if len(grad) != len(theta) {
			return fmt.Errorf("grad size mismatch: got %d, want %d", len(grad), len(theta))
		}
		for i := range theta {
			theta[i] -= lrm.LearningRate * grad[i]
		}

		loss := lrm.Method.CalcLoss(Xb, y, theta)
		lrm.History = append(lrm.History, loss)
	}

	lrm.Theta = theta
	return nil
}

// Predict возвращает прогнозы для X (без колонки единиц).
func (lrm *LinearRegression) Predict(X [][]float64) ([]float64, error) {
	if lrm.Theta == nil {
		return nil, fmt.Errorf("model is not fitted")
	}
	Xb := addOnesColumn(X)
	return matVecMul(Xb, lrm.Theta), nil
}

// ----------------- Вспомогательные функции -----------------

func addOnesColumn(X [][]float64) [][]float64 {
	m := len(X)
	n := 0
	if m > 0 {
		n = len(X[0])
	}
	Xb := make([][]float64, m)
	for i := 0; i < m; i++ {
		row := make([]float64, n+1)
		row[0] = 1.0
		copy(row[1:], X[i])
		Xb[i] = row
	}
	return Xb
}

func matVecMul(X [][]float64, v []float64) []float64 {
	m := len(X)
	res := make([]float64, m)
	for i := 0; i < m; i++ {
		sum := 0.0
		row := X[i]
		for j := 0; j < len(row) && j < len(v); j++ {
			sum += row[j] * v[j]
		}
		res[i] = sum
	}
	return res
}

func transposeTimesVec(X [][]float64, vec []float64) []float64 {
	// возвращает X^T * vec (без усреднения)
	if len(X) == 0 {
		return nil
	}
	ncols := len(X[0])
	res := make([]float64, ncols)
	for j := 0; j < ncols; j++ {
		sum := 0.0
		for i := 0; i < len(X); i++ {
			sum += X[i][j] * vec[i]
		}
		res[j] = sum
	}
	return res
}

// ----------------- Простая реализация пакетного GD -----------------

// BatchGD — пример реализации GDMethod (пакетный градиентный спуск).
type BatchGD struct{}

func (b *BatchGD) CalcLoss(Xb [][]float64, y []float64, theta []float64) float64 {
	preds := matVecMul(Xb, theta)
	sum := 0.0
	for i := 0; i < len(preds); i++ {
		diff := preds[i] - y[i]
		sum += diff * diff
	}
	mean := sum / float64(len(preds))
	return 0.5 * mean
}

func (b *BatchGD) CalcGrad(Xb [][]float64, y []float64, theta []float64) []float64 {
	m := len(Xb)
	preds := matVecMul(Xb, theta)
	err := make([]float64, m)
	for i := 0; i < m; i++ {
		err[i] = preds[i] - y[i]
	}
	grad := transposeTimesVec(Xb, err)
	// усредним градиент
	for i := range grad {
		grad[i] /= float64(m)
	}
	return grad
}

// ----------------- Дополнительная полезная функция: R2 -----------------

func RSquared(yTrue, yPred []float64) float64 {
	if len(yTrue) != len(yPred) || len(yTrue) == 0 {
		return math.NaN()
	}
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		ssRes += d * d
		td := yTrue[i] - mean
		ssTot += td * td
	}
	if ssTot == 0 {
		return 0.0
	}
	return 1.0 - ssRes/ssTot
}
````

Кратко:
- GDMethod — абстрактный интерфейс с CalcLoss и CalcGrad.
- LinearRegression принимает реализацию GDMethod и использует её в Fit (обновление theta в общем цикле).
- В пакете есть пример BatchGD; при необходимости добавлю SGD / MiniBatchGD реализующие тот же интерфейс.

IgorKalet: Перепиши это на python с numpy

GitHub Copilot: Ниже — портирование на Python с использованием numpy. Абстрактный класс GDMethod с методами calc_loss и calc_grad, общий класс LinearRegression, и пример BatchGD.

````python
import abc
from typing import Any
import numpy as np


class GDMethod(abc.ABC):
    """
    Интерфейс для метода градиентного спуска.

    Методы получают X_b (матрица с колонкой единиц), y и текущие theta.
    calc_grad должен возвращать усреднённый по батчу градиент (т.е. уже делённый на m),
    чтобы шаг обновления был просто theta -= lr * grad.
    """

    @abc.abstractmethod
    def calc_loss(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_grad(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BatchGD(GDMethod):
    """Простая реализация пакетного градиентного спуска."""

    def calc_loss(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        preds = X_b.dot(theta)
        mse = np.mean((preds - y) ** 2)
        return 0.5 * mse

    def calc_grad(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        m = X_b.shape[0]
        preds = X_b.dot(theta)
        err = preds - y
        grad = X_b.T.dot(err) / float(m)
        return grad


class LinearRegression:
    """
    Общий класс линейной регрессии, принимает реализацию GDMethod.
    """

    def __init__(self, method: GDMethod, lr: float = 0.01, epochs: int = 1000):
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
    # Пример использования
    rng = np.random.default_rng(0)
    X = 2 * rng.random((200, 2))
    true_coef = np.array([3.0, -1.5])
    true_intercept = 2.0
    y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    method = BatchGD()
    model = LinearRegression(method=method, lr=0.1, epochs=300)
    model.fit(X, y)
    preds = model.predict(X)
    print("theta:", model.theta)
    print("R2:", model.r2_score(X, y))
````

IgorKalet: Реализуй в данном коде стохостический градиентный спуск

GitHub Copilot: Добавлен класс SGD (стоxастический градиентный спуск). Он выбирает случайный сэмпл при каждом вызове calc_grad и возвращает градиент по этому сэмплу (т.е. стохастический шаг). Пример использования — передать экземпляр SGD в LinearRegression.

````python
import abc
from typing import Any
import numpy as np


class GDMethod(abc.ABC):
    """
    Интерфейс для метода градиентного спуска.

    Методы получают X_b (матрица с колонкой единиц), y и текущие theta.
    calc_grad должен возвращать усреднённый по батчу градиент (т.е. уже делённый на m),
    чтобы шаг обновления был просто theta -= lr * grad.
    """

    @abc.abstractmethod
    def calc_loss(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_grad(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BatchGD(GDMethod):
    """Простая реализация пакетного градиентного спуска."""

    def calc_loss(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        preds = X_b.dot(theta)
        mse = np.mean((preds - y) ** 2)
        return 0.5 * mse

    def calc_grad(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        m = X_b.shape[0]
        preds = X_b.dot(theta)
        err = preds - y
        grad = X_b.T.dot(err) / float(m)
        return grad


class SGD(GDMethod):
    """
    Стохастический градиентный спуск.
    Каждый вызов calc_grad выбирает один случайный пример и возвращает градиент по нему.
    Это даёт стохастические обновления, если в цикле fit вызывается calc_grad каждую итерацию.
    Опционально можно передать rng (numpy Generator) для детерминированности.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def calc_loss(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        # для оценки loss используем полный набор
        preds = X_b.dot(theta)
        mse = np.mean((preds - y) ** 2)
        return 0.5 * mse

    def calc_grad(self, X_b: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        m = X_b.shape[0]
        if m == 0:
            return np.zeros_like(theta)
        idx = int(self.rng.integers(0, m))
        Xi = X_b[idx:idx+1]        # shape (1, n+1)
        yi = y[idx:idx+1]         # shape (1,)
        pred = Xi.dot(theta)      # shape (1,)
        err = pred - yi           # shape (1,)
        grad = Xi.T.dot(err)      # shape (n+1, 1)
        # grad для одного примера — усреднять на 1 не нужно, возвращаем одномерный вектор
        return grad.ravel()


class LinearRegression:
    """
    Общий класс линейной регрессии, принимает реализацию GDMethod.
    """

    def __init__(self, method: GDMethod, lr: float = 0.01, epochs: int = 1000):
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
    batch_method = BatchGD()
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
````

IgorKalet: объясни подгобнее что делает метод  _add_bias

GitHub Copilot: Метод _add_bias добавляет столбец единиц в начало матрицы признаков X — это делается, чтобы вектор параметров theta содержал свободный член (intercept) в позиции theta[0]. Кратко по шагам:

- Приводит X к numpy-массиву с типом float.
- Если X — 1D (вектор признака), превращает в 2D столбец (m,1).
- Создаёт массив единиц формы (m, 1).
- Соединяет единицы и исходные признаки по горизонтали (np.hstack), в результате получается X_b с формой (m, n+1), где первая колонка — единицы.

Почему это важно:
- Линейная модель обычно имеет вид y = theta0 + theta1*x1 + ...; колонка единиц позволяет учесть theta0 как обычный параметр в векторе theta и писать прогноз как X_b.dot(theta).
- Без добавления единиц модель принуждала бы прямую проходить через 0 (no intercept).

Пример (покажет трансформацию):
````python
# python
import numpy as np

def add_bias(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    m = X.shape[0]
    return np.hstack([np.ones((m, 1)), X])

X = np.array([[2.0, 3.0],
              [4.0, 5.0],
              [6.0, 7.0]])
print(add_bias(X))
# Результат:
# [[1. 2. 3.]
#  [1. 4. 5.]
#  [1. 6. 7.]]
````
Замечание: если вы уже добавили колонку единиц вручную, не вызывайте _add_bias снова — получится лишняя колонка единиц.

IgorKalet: объясни подгобнее что делает метод  r2_score

GitHub Copilot: Метод r2_score вычисляет коэффициент детерминации R² — меру того, какая доля дисперсии целевой переменной объясняется моделью. Кратко шаги и смысл:

1. Получает предсказания модели: preds = self.predict(X).
2. Вычисляет сумму квадратов остатков (SS_res):
   SS_res = sum((y_i - preds_i)^2). Это несоответствие модели данным.
3. Вычисляет общую сумму квадратов относительно среднего (SS_tot):
   SS_tot = sum((y_i - mean(y))^2). Это дисперсия истинных значений.
4. Возвращает R² = 1 - SS_res / SS_tot при SS_tot != 0. Если SS_tot == 0 (все y одинаковы), метод возвращает 0.0 (в коде так сделано, чтобы не делить на ноль).

Интерпретация:
- R² = 1.0 — модель предсказывает идеально (SS_res = 0).
- R² = 0.0 — модель объясняет столько же дисперсии, сколько простая константная модель y = mean(y).
- R² < 0.0 — модель хуже, чем предсказание константой (SS_res > SS_tot). Может быть большим по модулю при очень плохой модели.

Формулы (в коде реализованы численно):
- mean_y = (1/N) * Σ y_i
- SS_res = Σ (y_i - preds_i)^2
- SS_tot = Σ (y_i - mean_y)^2
- R² = 1 - SS_res / SS_tot  (если SS_tot != 0, иначе 0.0)

Небольшой численный пример:
````python
# python
y = [3.0, 4.0, 5.0]
preds = [2.8, 4.1, 5.2]
mean_y = (3+4+5)/3 = 4.0
SS_res = (3-2.8)**2 + (4-4.1)**2 + (5-5.2)**2 = 0.04 + 0.01 + 0.04 = 0.09
SS_tot = (3-4)**2 + (4-4)**2 + (5-4)**2 = 1 + 0 + 1 = 2
R2 = 1 - 0.09 / 2 = 0.955
````

Замечания:
- Поведение при константном y (SS_tot == 0) может быть изменено (например, вернуть NaN или 1.0 в зависимости от определения). Текущее поведение возвращает 0.0.
- R² чувствителен к масштабам и выбросам; для несопоставимых моделей лучше смотреть дополнительные метрики (MAE, RMSE).

IgorKalet: Как правило, в начале спуска мы хотим делать большие шаги, чтобы поскорее подойти поближе к минимуму, а позже мы уже хотим делать шаги маленькие, чтобы более точнее этого минимума достичь и не "перепрыгнуть" его. Как это дечше сделать.

GitHub Copilot: Copilot took too long to get ready. Please ensure you are signed in to GitHub and that the extension `GitHub.copilot-chat` is installed and enabled.

IgorKalet: Как правило, в начале спуска мы хотим делать большие шаги, чтобы поскорее подойти поближе к минимуму, а позже мы уже хотим делать шаги маленькие, чтобы более точнее этого минимума достичь и не "перепрыгнуть" его. Как это дечше сделать.

GitHub Copilot: Кратко: добавил набор планировщиков шага (learning rate schedulers) и показал, как их применять — можно передать scheduler в LinearRegression или в функции gradient_descent / stochastic_gradient_descent. Scheduler — callable, принимающий номер шага (iteration) и возвращающий float lr. Рекомендации: использовать быстрый lr в начале (warmup/large lr) и затем экспоненциальное / инверсное уменьшение; для стабильности на стохастике полезен небольшой lr и/или адаптивные оптимизаторы (Adam).

Ниже — предложенные изменения в файле regression.py.

````python
# ...existing code...
import abc
from typing import List, Callable
import numpy as np

# ----------------- new: learning rate schedulers -----------------
class LRScheduler:
    """Базовый класс/интерфейс: объект должен быть вызываемым и возвращать lr по шагу."""
    def __call__(self, step: int) -> float:
        raise NotImplementedError

class ConstantLR(LRScheduler):
    def __init__(self, lr: float):
        self.lr = float(lr)
    def __call__(self, step: int) -> float:
        return self.lr

class TimeDecay(LRScheduler):
    # lr = lr0 / (1 + decay * step)
    def __init__(self, lr0: float, decay: float):
        self.lr0 = float(lr0); self.decay = float(decay)
    def __call__(self, step: int) -> float:
        return self.lr0 / (1.0 + self.decay * float(step))

class ExponentialDecay(LRScheduler):
    # lr = lr0 * exp(-k * step)
    def __init__(self, lr0: float, k: float):
        self.lr0 = float(lr0); self.k = float(k)
    def __call__(self, step: int) -> float:
        return self.lr0 * np.exp(-self.k * float(step))

class StepDecay(LRScheduler):
    # lr = lr0 * drop_rate^(floor(step / step_size))
    def __init__(self, lr0: float, drop_rate: float = 0.5, step_size: int = 100):
        self.lr0 = float(lr0); self.drop_rate = float(drop_rate); self.step_size = int(step_size)
    def __call__(self, step: int) -> float:
        return self.lr0 * (self.drop_rate ** (step // self.step_size))

class InverseScaling(LRScheduler):
    # lr = lr0 / (1 + k * step) ** power
    def __init__(self, lr0: float, k: float = 1e-3, power: float = 0.5):
        self.lr0 = float(lr0); self.k = float(k); self.power = float(power)
    def __call__(self, step: int) -> float:
        return self.lr0 / ((1.0 + self.k * float(step)) ** self.power)

class CosineAnnealing(LRScheduler):
    # cosine annealing with period T and min lr
    def __init__(self, lr0: float, lr_min: float, T: int):
        self.lr0 = float(lr0); self.lr_min = float(lr_min); self.T = int(T)
    def __call__(self, step: int) -> float:
        if self.T <= 0:
            return self.lr0
        x = (step % self.T) / float(self.T)
        return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * (1.0 + np.cos(np.pi * x))

# helper to accept either float or scheduler
def _make_scheduler(lr_or_sched: float | Callable[[int], float]) -> Callable[[int], float]:
    if callable(lr_or_sched):
        return lr_or_sched
    return ConstantLR(float(lr_or_sched))

# ...existing code...

class BaseLoss(abc.ABC):
    @abc.abstractmethod
    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# ...existing code...
class LinearRegression:
    """
    Общий класс линейной регрессии, принимает реализацию GDMethod.
    """

    def __init__(self, method: BaseLoss, lr: float = 0.01, epochs: int = 1000, lr_scheduler: Callable[[int], float] | None = None):
        self.method = method
        self.lr = float(lr)
        self.epochs = int(epochs)
        # lr_scheduler может быть callable(step)->lr или None
        self.lr_scheduler = _make_scheduler(lr_scheduler) if lr_scheduler is not None else ConstantLR(self.lr)
        self.theta: np.ndarray | None = None
        self.history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        X_b = add_bias(X)
        n_params = X_b.shape[1]
        theta = np.zeros(n_params, dtype=float)

        self.history = []
        for epoch in range(self.epochs):
            lr = float(self.lr_scheduler(epoch))
            grad = self.method.calc_grad(X_b, y, theta)
            if grad.shape != theta.shape:
                raise ValueError("gradient shape mismatch")
            theta = theta - lr * grad

            loss = float(self.method.calc_loss(X_b, y, theta))
            self.history.append(loss)

        self.theta = theta
        return self

# ...existing code...

def gradient_descent(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    loss: BaseLoss,
    lr: float | Callable[[int], float],
    n_iterations: int = 100000,
) -> List[np.ndarray]:
    """
    Функция градиентного спуска (детерминированный / batch).
    lr может быть константой или callable(step)->lr.
    """
    scheduler = _make_scheduler(lr)
    history = []
    w = w_init.copy()
    for it in range(n_iterations):
        current_lr = float(scheduler(it))
        grad = loss.calc_grad(X, y, w)
        w = w - current_lr * grad
        history.append(w.copy())

    return history

def stochastic_gradient_descent(
    w_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    loss: BaseLoss,
    lr: float | Callable[[int], float],
    batch_size: int,
    n_iterations: int = 1000,
) -> List[np.ndarray]:
    """
    Стохастический / мини-батч. lr может быть callable(step)->lr.
    """
    scheduler = _make_scheduler(lr)
    history = []
    w = w_init.copy()
    m = X.shape[0]
    for it in range(n_iterations):
        current_lr = float(scheduler(it))
        batch_indices = np.random.choice(m, size=batch_size, replace=False)
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        grad = loss.calc_grad(batch_X, batch_y, w)
        w = w - current_lr * grad
        history.append(w.copy())

    return history

# ...existing code...
if __name__ == "__main__":
    # пример использования scheduler
    rng = np.random.default_rng(0)
    X = 2 * rng.random((200, 2))
    true_coef = np.array([3.0, -1.5])
    true_intercept = 2.0
    y = true_intercept + X.dot(true_coef) + rng.normal(0, 0.5, size=200)

    loss = MSELoss()

    # экспоненциальный спад
    exp_sched = ExponentialDecay(lr0=0.1, k=0.01)
    model = LinearRegression(method=loss, lr=0.1, epochs=300, lr_scheduler=exp_sched)
    model.fit(X, y)
    print("theta:", model.theta)
    print("loss start/stop:", model.history[0], model.history[-1])

    # стохастический градиент с инверсным масштабированием lr
    inv_sched = InverseScaling(lr0=0.05, k=1e-3, power=0.6)
    w_init = np.random.uniform(-2, 2, (X.shape[1] + 1,))
    history = stochastic_gradient_descent(w_init, add_bias(X), y, loss, lr=inv_sched, batch_size=16, n_iterations=1000)
    print("sgd final loss:", loss.calc_loss(add_bias(X).dot(np.eye(add_bias(X).shape[1])), y, history[-1]))  # демонстрация
````

Короткие практические советы:
- Для классического подхода: warmup (первые 5–10 шагов lr растёт/фиксируется), затем экспоненциальный или inverse decay.
- Для SGD/мини-батч часто используют inverse scaling или step decay.
- Для быстрой и стабильной сходимости в большинстве задач — Adam или RMSProp (адаптивные алгоритмы) вместо вручную настроенных lr.
- Не задавайте слишком резкое уменьшение lr в начале — это замедлит прогресс; уменьшайте плавно.

Если хотите, могу:
- Добавить warmup scheduler.
- Добавить Adam/RMSProp реализации и интегрировать в BaseLoss/методику.
- Внести patch прямо в ваш файл (уже предложил изменения — скажите, применять).

IgorKalet: Как посчитать ошибку обученной регрессии на обучающей и тестовой выборке при помощи метода mean_squared_error из sklearn.metrics

GitHub Copilot: Коротко: импортируете mean_squared_error, делаете predict на train и test и считаете MSE. Пример — вставьте в ваш скрипт:

````python
# python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# предположим, X, y — ваши данные; y может быть (n,1) или (n,)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(loss= MSELoss(), lr=0.1)  # или ваша настроенная модель
model.fit(X_train, y_train)

# предсказания (убедитесь, что y_true и y_pred имеют одинаковую форму)
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# если y имеют форму (n,1), привести к (n,)
y_train_true = np.asarray(y_train).reshape(-1)
y_test_true  = np.asarray(y_test).reshape(-1)
y_train_pred = np.asarray(y_train_pred).reshape(-1)
y_test_pred  = np.asarray(y_test_pred).reshape(-1)

mse_train = mean_squared_error(y_train_true, y_train_pred)
mse_test  = mean_squared_error(y_test_true,  y_test_pred)

print("MSE train:", mse_train)
print("MSE test: ", mse_test)

# при желании RMSE:
rmse_test = np.sqrt(mse_test)
print("RMSE test:", rmse_test)
````

Замечания:
- mean_squared_error по умолчанию возвращает среднюю квадратичную ошибку (squared=True).
- Для воспроизводимости фиксируйте random_state в train_test_split.
- Убедитесь, что model.predict возвращает одномерный массив (shape (n,)), иначе приведите reshape(-1).

IgorKalet: Реализовать градиентный спуск на основе MSE с L2 регуляризацией

GitHub Copilot: Добавил класс MSELossL2 с L2-регуляризацией (ridge). Регуляризация по умолчанию не включает bias (первый элемент w), коэффициент регуляризации — alpha, в loss добавляется 0.5 * alpha * ||w_reg||^2, в градиент — alpha * w_reg.

````python
# ...existing code...
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

# ...existing code...

class MSELossL2(BaseLoss):
    """
    MSE + L2 (ridge) regularization.
    loss = MSE + 0.5 * alpha * ||w_reg||^2
    grad = grad_MSE + alpha * w_reg
    По умолчанию bias (первый компонент w) не регуляризуется (reg_bias=False).
    """
    def __init__(self, alpha: float = 1e-3, reg_bias: bool = False):
        self.alpha = float(alpha)
        self.reg_bias = bool(reg_bias)

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        mse = np.mean((preds - y) ** 2)
        if self.alpha == 0.0:
            return mse
        w_reg = w.copy()
        if not self.reg_bias and w_reg.size > 0:
            w_reg[0] = 0.0
        reg_term = 0.5 * self.alpha * np.sum(w_reg ** 2)
        return mse + reg_term

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        preds = X.dot(w)
        err = preds - y
        grad = 2 * X.T.dot(err) / float(m)
        if self.alpha != 0.0:
            w_reg = w.copy()
            if not self.reg_bias and w_reg.size > 0:
                w_reg[0] = 0.0
            grad = grad + self.alpha * w_reg
        return grad
# ...existing code...
````

Пример использования:
- Передать loss = MSELossL2(alpha=0.01) в LinearRegression для обучения с L2.

IgorKalet: как с помощью mean_squared_error  можно оценить качество обучения модели

GitHub Copilot: Коротко — mean_squared_error (sklearn.metrics) возвращает средний квадрат ошибки (MSE). Чем меньше MSE (и RMSE = sqrt(MSE)), тем лучше модель. Обычно считают и для обучающей, и для тестовой выборки, сравнивают (overfit если train << test). Пример использования:

````python
# python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# X, y — ваши данные; model — обученная модель с методом predict(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)  # обучили модель

y_train_pred = model.predict(X_train).reshape(-1)
y_test_pred  = model.predict(X_test).reshape(-1)

y_train_true = np.asarray(y_train).reshape(-1)
y_test_true  = np.asarray(y_test).reshape(-1)

mse_train = mean_squared_error(y_train_true, y_train_pred)
mse_test  = mean_squared_error(y_test_true,  y_test_pred)

rmse_train = np.sqrt(mse_train)
rmse_test  = np.sqrt(mse_test)

print("MSE train:", mse_train, "RMSE train:", rmse_train)
print("MSE test: ", mse_test,  "RMSE test: ", rmse_test)
````

Советы по интерпретации:
- RMSE в тех же единицах, что и y — удобно для понимания ошибки.
- Сравнивайте train и test: большое отличие → возможен overfitting.
- Для относительной оценки используйте Normalized RMSE (RMSE / (y.max()-y.min())) или RMSE / y.std().
- Для оценки на k-fold кросс-валидации используйте sklearn.model_selection.cross_val_score с scoring='neg_mean_squared_error' (возвращает отрицательное MSE — берите минус и sqrt для RMSE).

IgorKalet: Реализовать градиентный спуск на основе Huber Loss

GitHub Copilot: ````python
# ...existing code...
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

class HuberLoss(BaseLoss):
    """
    Huber loss (smooth L1). Parameter delta controls transition point:
      L_delta(r) = 0.5 * r^2                 if |r| <= delta
                 = delta * (|r| - 0.5*delta) if |r| >  delta
    Loss returned is the average over samples.
    Gradient is computed w.r.t. parameters w: grad = X^T * dL/dr / m,
    where dL/dr = r (if |r|<=delta) or delta * sign(r) (otherwise).
    Expects X to have bias column if you want bias in w.
    """
    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        r = preds - y
        abs_r = np.abs(r)
        # vectorized huber per-sample
        small_mask = abs_r <= self.delta
        loss_vals = np.empty_like(r, dtype=float)
        loss_vals[small_mask] = 0.5 * (r[small_mask] ** 2)
        loss_vals[~small_mask] = self.delta * (abs_r[~small_mask] - 0.5 * self.delta)
        return float(np.mean(loss_vals))

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        if m == 0:
            return np.zeros_like(w)
        preds = X.dot(w)
        r = preds - y
        # derivative dL/dr
        grad_r = np.where(np.abs(r) <= self.delta, r, self.delta * np.sign(r))
        grad = X.T.dot(grad_r) / float(m)
        return grad
# ...existing code...
```# filepath: d:\test\git_my\sudy_ml\regression.py
# ...existing code...
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

class HuberLoss(BaseLoss):
    """
    Huber loss (smooth L1). Parameter delta controls transition point:
      L_delta(r) = 0.5 * r^2                 if |r| <= delta
                 = delta * (|r| - 0.5*delta) if |r| >  delta
    Loss returned is the average over samples.
    Gradient is computed w.r.t. parameters w: grad = X^T * dL/dr / m,
    where dL/dr = r (if |r|<=delta) or delta * sign(r) (otherwise).
    Expects X to have bias column if you want bias in w.
    """
    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        r = preds - y
        abs_r = np.abs(r)
        # vectorized huber per-sample
        small_mask = abs_r <= self.delta
        loss_vals = np.empty_like(r, dtype=float)
        loss_vals[small_mask] = 0.5 * (r[small_mask] ** 2)
        loss_vals[~small_mask] = self.delta * (abs_r[~small_mask] - 0.5 * self.delta)
        return float(np.mean(loss_vals))

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        if m == 0:
            return np.zeros_like(w)
        preds = X.dot(w)
        r = preds - y
        # derivative dL/dr
        grad_r = np.where(np.abs(r) <= self.delta, r, self.delta * np.sign(r))
        grad = X.T.dot(grad_r) / float(m)
        return grad
# ...existing code...
