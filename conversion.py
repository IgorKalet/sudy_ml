import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from regression import BaseLoss, MSELoss, LinearRegression

X_raw = pd.read_csv(
    "data.csv",
    header=None,
    na_values=["?"],
)

X_raw = X_raw[~X_raw[25].isna()].reset_index()

y = X_raw[25]
X_raw = X_raw.drop(25, axis=1)

# Для воспроизводимости
np.random.seed(42)

shuffled_indices = np.random.permutation(len(X_raw))
test_set_size = int(len(X_raw) * 0.2)

test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

train_X = X_raw.loc[train_indices]
test_X = X_raw.loc[test_indices]

train_y = y.loc[train_indices]
test_y = y.loc[test_indices]

numerical_cols = train_X.select_dtypes(include=np.number).columns
categorical_cols = train_X.select_dtypes(include='object').columns

# --- Заполнение для числовых признаков (медианой) ---
for col in numerical_cols:
    median_val = train_X[col].median()
    train_X[col] = train_X[col].fillna(median_val)
    test_X[col] = test_X[col].fillna(median_val)

# --- Заполнение для категориальных признаков (модой) ---
for col in categorical_cols:
    mode_val = train_X[col].mode()[0]
    train_X[col] = train_X[col].fillna(mode_val)
    test_X[col] = test_X[col].fillna(mode_val)

train_X_cat_dummies = pd.get_dummies(train_X[categorical_cols], prefix=categorical_cols)
test_X_cat_dummies = pd.get_dummies(test_X[categorical_cols], prefix=categorical_cols)

train_cat_cols = train_X_cat_dummies.columns
test_X_cat_dummies = test_X_cat_dummies.reindex(columns=train_cat_cols, fill_value=0)

# Теперь удаляем старые категориальные столбцы и добавляем новые
train_X_processed = train_X.drop(categorical_cols, axis=1)
test_X_processed = test_X.drop(categorical_cols, axis=1)

train_X_processed = pd.concat([train_X_processed, train_X_cat_dummies], axis=1)
test_X_processed = pd.concat([test_X_processed, test_X_cat_dummies], axis=1)

for col in numerical_cols:
    # Рассчитываем среднее и стандартное отклонение на обучающей выборке
    mean_val = train_X_processed[col].mean()
    std_val = train_X_processed[col].std()

    # Применяем стандартизацию к обеим выборкам
    train_X_processed[col] = (train_X_processed[col] - mean_val) / std_val
    test_X_processed[col] = (test_X_processed[col] - mean_val) / std_val

def fitModel(loss: BaseLoss, caption: str) -> LinearRegression:
    model = LinearRegression(loss=loss, lr=0.01)
    model.fit(train_X_processed, train_y)

    y_train_pred = model.predict(train_X_processed)
    y_test_pred  = model.predict(test_X_processed)

    mse_train = mean_squared_error(y_train_pred, train_y)
    mse_test  = mean_squared_error(y_test_pred,  test_y)

    print(f"MSE {caption} train:", mse_train)
    print(f"MSE {caption} test: ", mse_test)
    print(f"MSE {caption} r2: ", model.r2_score(test_X_processed, test_y))

    return model


loss = MSELoss()
fitModel(loss, 'simple')

class MSEL2Loss(BaseLoss):
    def __init__(self, coef: float = 1.0):
        self.coef = coef

    def calc_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        preds = X.dot(w)
        mse = np.mean((preds - y) ** 2)
        w_reg = w.copy()
        if w_reg.size > 0:
            w_reg[0] = 0.0
        reg_term = self.coef * np.sum(w_reg ** 2)
        return mse + reg_term

    def calc_grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        preds = X.dot(w)
        err = preds - y
        grad = 2 * X.T.dot(err) / float(m)
        w_reg = w.copy()
        if w_reg.size > 0:
            w_reg[0] = 0.0
        grad = grad + 2 * self.coef * w_reg
        return grad

loss = MSEL2Loss(0.1)
fitModel(loss, 'L2 (0.1)')
loss = MSEL2Loss(0.005)
fitModel(loss, 'L2 (0.005)')

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

loss = HuberLoss(30)
modelHL = fitModel(loss, 'HL (1)')
