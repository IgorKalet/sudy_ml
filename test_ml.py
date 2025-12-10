import math

# task 1
def calculate_entropy(class_counts: dict) -> float:
    total_samples = sum(class_counts.values())

    if total_samples == 0:
        return 0.0

    entropy = 0.0

    for count in class_counts.values():
        probability = count / total_samples
        entropy += probability * math.log2(probability)

    return -entropy

node_data = {'k1': 8, 'k2': 2}
entropy_value = calculate_entropy(node_data)

print(f"Энтропия: {entropy_value:.2f}")


# task 2
def calculate_gini(class_counts: dict) -> float:
    total_samples = sum(class_counts.values())

    if total_samples == 0:
        return 0.0

    gini = 0.0

    for count in class_counts.values():
        if total_samples > 0:
            probability = count / total_samples
            gini += probability * (1 - probability)

    return gini

def calculate_information_gain_gini(parent_counts: dict, left_counts: dict, right_counts: dict) -> float:
    n_parent = sum(parent_counts.values())
    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())

    if n_parent == 0:
        return 0.0

    gini_parent = calculate_gini(parent_counts)
    gini_left = calculate_gini(left_counts)
    gini_right = calculate_gini(right_counts)

    weighted_gini_children = (n_left / n_parent) * gini_left + (n_right / n_parent) * gini_right
    information_gain = gini_parent - weighted_gini_children

    return information_gain

parent_node_counts = {'k1': 8, 'k2': 2}
left_node_counts = {'k1': 8}
right_node_counts = {'k2': 2}

gain_value = calculate_information_gain_gini(parent_node_counts, left_node_counts, right_node_counts)

print(f"Критерий информативности: {gain_value:.2f}")

# task 3
def predict_regression_leaf(target_values: list) -> float:
    if not target_values:
        return 0.0

    total_sum = sum(target_values)
    count = len(target_values)

    return total_sum / count

leaf_target_values = [1, 10, 5, 18, 100, 30, 50, 61, 84, 47]
prediction = predict_regression_leaf(leaf_target_values)

print(f"Предсказание модели: {prediction}")


import numpy as np
import pandas as pd
from typing import Tuple, Union

def find_best_split(
    feature_vector: Union[np.ndarray, pd.Series],
    target_vector: Union[np.ndarray, pd.Series],
    task: str = "classification",
    feature_type: str = "real"
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Находит оптимальное разбиение для одного признака.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    * Поведение функции в случае константного признака возвращает пустые массивы и -np.inf.
    * При одинаковых приростах Джини или дисперсии нужно выбирать минимальный сплит.
    * Реализация для числовых признаков полностью векторизована (без циклов).

    :param feature_vector: вещественнозначный вектор значений признака.
    :param target_vector: вектор классов или значений, len(feature_vector) == len(target_vector).
    :param task: либо `classification`, либо `regression`.
    :param feature_type: либо `real`, либо `categorical`.

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами.
    :return impurities: вектор со значениями критерия неопределенности для каждого порога.
    :return threshold_best: оптимальный порог (число).
    :return impurity_best: оптимальное значение критерия неопределенности (число).
    """
    # 1. Предобработка и подготовка
    x = np.asarray(feature_vector)
    y = np.asarray(target_vector)

    # Проверка на корректность входных данных
    if x.shape[0] != y.shape[0] or x.shape[0] == 0:
        return np.array([]), np.array([]), -np.inf, -np.inf

    # 2. Выбор функции для расчета неопределенности
    if task == "classification":
        # Для классификации используем Джини
        impurity_func = lambda y_sub: 1.0 - np.sum((np.bincount(y_sub) / len(y_sub))**2)
    elif task == "regression":
        # Для регрессии используем дисперсию
        impurity_func = np.var
    else:
        raise ValueError("task должен быть 'classification' или 'regression'")

    # 3. Логика для числовых признаков (полностью векторизована)
    if feature_type == "real":
        # Сортируем оба вектора по значениям признака
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Находим уникальные значения и пороги
        unique_x = np.unique(x_sorted)
        if len(unique_x) <= 1:
            return np.array([]), np.array([]), -np.inf, -np.inf

        thresholds = (unique_x[:-1] + unique_x[1:]) / 2

        # --- Векторизованный расчет прироста информации ---
        # Используем кумулятивную сумму для быстрого расчета статистик по подвыборкам

        # Преобразуем y в целые числа для np.bincount, если это классификация
        if task == "classification":
             _, y_sorted = np.unique(y_sorted, return_inverse=True)

        # Количество объектов в левой подвыборке для каждого возможного разбиения
        n_left = np.arange(1, y_sorted.shape[0])
        n_right = y_sorted.shape[0] - n_left

        # Кумулятивные суммы для левой подвыборки
        # Для регрессии нужны суммы и суммы квадратов
        if task == "regression":
            sum_left = np.cumsum(y_sorted)[:-1]
            sum_sq_left = np.cumsum(y_sorted**2)[:-1]

            # Статистики для правой подвыборки
            total_sum = sum_left[-1]
            total_sum_sq = sum_sq_left[-1]
            sum_right = total_sum - sum_left
            sum_sq_right = total_sum_sq - sum_sq_left

            # Дисперсия для левой и правой подвыборок
            var_left = (sum_sq_left / n_left) - (sum_left / n_left)**2
            var_right = (sum_sq_right / n_right) - (sum_right / n_right)**2

            impurity_left = var_left
            impurity_right = var_right
        else: # classification
            # Создаем матрицу one-hot encoding
            n_classes = len(np.unique(y_sorted))
            one_hot = np.zeros((y_sorted.shape[0], n_classes), dtype=int)
            one_hot[np.arange(y_sorted.shape[0]), y_sorted] = 1

            # Кумулятивные суммы для подсчета классов в левой подвыборке
            class_counts_left = np.cumsum(one_hot, axis=0)[:-1]

            # Счетчики классов для правой подвыборки
            total_class_counts = class_counts_left[-1:]
            class_counts_right = total_class_counts - class_counts_left

            # Расчет Джини
            p_left = class_counts_left / n_left[:, np.newaxis]
            p_right = class_counts_right / n_right[:, np.newaxis]

            impurity_left = 1.0 - np.sum(p_left**2, axis=1)
            impurity_right = 1.0 - np.sum(p_right**2, axis=1)

        # Неопределенность родительской вершины
        impurity_parent = impurity_func(y_sorted)

        # Прирост информации для каждого порога
        gain = impurity_parent - (n_left / y_sorted.shape[0]) * impurity_left - (n_right / y_sorted.shape[0]) * impurity_right

        # Нам нужно соответствие между порогами и приростами
        # Уникальные значения в x_sorted могут повторяться, нужно сгруппировать приросты
        # Найдем индексы, где x_sorted меняет значение
        split_idx = np.where(np.diff(x_sorted) > 0)[0]

        # Усредняем прирост для одинаковых порогов
        # Используем np.add.reduceat для эффективной группировки
        # Создаем массив индексов для редукции
        reduceat_idx = np.concatenate(([0], split_idx + 1))

        # Количество элементов в каждой группе
        counts_in_groups = np.diff(np.concatenate((reduceat_idx, [y_sorted.shape[0]])))

        # Суммируем приросты по группам и делим на их количество
        grouped_gain = np.add.reduceat(gain, reduceat_idx) / counts_in_groups

        # Отфильтровываем разбиения, создающие пустые поддеревья (это уже учтено в n_left/n_right)
        # и соответствуем отфильтрованным порогам
        final_thresholds = thresholds
        final_impurities = grouped_gain

    # 4. Логика для категориальных признаков (наивный подход)
    elif feature_type == "categorical":
        categories = np.unique(x)
        if len(categories) <= 1:
            return np.array([]), np.array([]), -np.inf, -np.inf

        impurities = []
        thresholds = []

        # Примечание: для категориальных признаков векторизация сложнее,
        # цикл по уникальным категориям является стандартным и эффективным решением.
        for category in categories:
            mask = (x == category)
            y_left = y[mask]
            y_right = y[~mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            impurity_left = impurity_func(y_left)
            impurity_right = impurity_func(y_right)
            impurity_parent = impurity_func(y)

            n_left, n_right = len(y_left), len(y_right)
            n_total = len(y)

            gain = impurity_parent - (n_left / n_total) * impurity_left - (n_right / n_total) * impurity_right

            impurities.append(gain)
            thresholds.append(category)

        final_impurities = np.array(impurities)
        final_thresholds = np.array(thresholds)

    else:
        raise ValueError("feature_type должен быть 'real' или 'categorical'")

    # 5. Выбор лучшего разбиения
    if len(final_impurities) == 0:
        return np.array([]), np.array([]), -np.inf, -np.inf

    # Находим индекс максимального прироста
    # np.argmax вернет первый индекс с максимальным значением, что обеспечивает выбор минимального порога при равенстве
    best_idx = np.argmax(final_impurities)

    threshold_best = final_thresholds[best_idx]
    impurity_best = final_impurities[best_idx]

    return final_thresholds, final_impurities, threshold_best, impurity_best

def find_best_split1(
    feature_vector: Union[np.ndarray, pd.DataFrame],
    target_vector: Union[np.ndarray, pd.Series],
    task: str = "classification",
    feature_type: str = "real"
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Находит оптимальное разбиение по признаку.

    :param feature_vector: вектор значений признака
    :param target_vector: вектор целевых значений
    :param task: 'classification' или 'regression'
    :param feature_type: 'real' или 'categorical'

    :return thresholds: отсортированные пороги
    :return ginis: значения критерия для каждого порога
    :return threshold_best: оптимальный порог
    :return gini_best: оптимальное значение критерия
    """

    feature_vector = np.asarray(feature_vector).flatten()
    target_vector = np.asarray(target_vector).flatten()
    n_samples = len(feature_vector)

    # Вычисляем критерий хаотичности для родительского узла
    if task == "classification":
        # Джини для родителя
        unique_classes, class_counts = np.unique(target_vector, return_counts=True)
        p = class_counts / n_samples
        parent_criterion = np.sum(p * (1 - p))
    else:
        # Дисперсия для родителя
        parent_criterion = np.var(target_vector)

    if feature_type == "real":
        # Для вещественных признаков ищем пороги между соседними значениями
        sorted_indices = np.argsort(feature_vector)
        sorted_feature = feature_vector[sorted_indices]
        sorted_target = target_vector[sorted_indices]

        # Уникальные значения для определения порогов
        unique_values = np.unique(sorted_feature)

        if len(unique_values) <= 1:
            # Константный признак
            return np.array([]), np.array([]), np.inf, -np.inf

        # Пороги - середины между соседними уникальными значениями
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        # Векторизованное вычисление критерия для каждого порога
        ginis = np.zeros(len(thresholds))

        for idx, threshold in enumerate(thresholds):
            left_mask = feature_vector <= threshold
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            # Пропускаем пороги, которые приводят к пустым поддеревьям
            if n_left == 0 or n_right == 0:
                ginis[idx] = -np.inf
                continue

            if task == "classification":
                # Джини для левого поддерева
                left_target = target_vector[left_mask]
                unique_left, counts_left = np.unique(left_target, return_counts=True)
                p_left = counts_left / n_left
                gini_left = np.sum(p_left * (1 - p_left))

                # Джини для правого поддерева
                right_target = target_vector[right_mask]
                unique_right, counts_right = np.unique(right_target, return_counts=True)
                p_right = counts_right / n_right
                gini_right = np.sum(p_right * (1 - p_right))

                # Информативность (прирост Джини)
                ginis[idx] = parent_criterion - (n_left / n_samples) * gini_left - (n_right / n_samples) * gini_right
            else:
                # Дисперсия для регрессии
                left_target = target_vector[left_mask]
                right_target = target_vector[right_mask]
                var_left = np.var(left_target)
                var_right = np.var(right_target)

                # Информативность (прирост в уменьшении дисперсии)
                ginis[idx] = parent_criterion - (n_left / n_samples) * var_left - (n_right / n_samples) * var_right

    else:  # categorical
        # Для категориальных признаков ищем лучшую категорию для левого поддерева
        unique_categories = np.unique(feature_vector)

        if len(unique_categories) <= 1:
            return np.array([]), np.array([]), np.inf, -np.inf

        thresholds = unique_categories
        ginis = np.zeros(len(thresholds))

        for idx, category in enumerate(thresholds):
            left_mask = feature_vector == category
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            # Пропускаем пороги, которые приводят к пустым поддеревьям
            if n_left == 0 or n_right == 0:
                ginis[idx] = -np.inf
                continue

            if task == "classification":
                # Джини для левого поддерева
                left_target = target_vector[left_mask]
                unique_left, counts_left = np.unique(left_target, return_counts=True)
                p_left = counts_left / n_left
                gini_left = np.sum(p_left * (1 - p_left))

                # Джини для правого поддерева
                right_target = target_vector[right_mask]
                unique_right, counts_right = np.unique(right_target, return_counts=True)
                p_right = counts_right / n_right
                gini_right = np.sum(p_right * (1 - p_right))

                # Информативность
                ginis[idx] = parent_criterion - (n_left / n_samples) * gini_left - (n_right / n_samples) * gini_right
            else:
                # Дисперсия для регрессии
                left_target = target_vector[left_mask]
                right_target = target_vector[right_mask]
                var_left = np.var(left_target)
                var_right = np.var(right_target)

                # Информативность
                ginis[idx] = parent_criterion - (n_left / n_samples) * var_left - (n_right / n_samples) * var_right

    # Находим лучший порог
    valid_mask = ginis > -np.inf

    if not np.any(valid_mask):
        # Нет валидных разбиений
        return np.array([]), np.array([]), np.inf, -np.inf

    best_idx = np.argmax(ginis[valid_mask])
    # Корректируем индекс с учётом фильтра
    best_idx = np.where(valid_mask)[0][best_idx]

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = pd.DataFrame(data=data["data"], columns=data["feature_names"])
y = data["target"]
print("Наилучшее разбиение: ", find_best_split(X["MedInc"], y, task="regression", feature_type="real"))
print("Наилучшее разбиение 1: ", find_best_split1(X["MedInc"], y, task="regression", feature_type="real"))


