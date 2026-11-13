
"""Задание 3: гиперпараметры и Feature Engineering."""

import csv
import logging
from itertools import combinations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# НАСТРОЙКИ ПРОЕКТА
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for folder in (PLOTS_DIR, MODELS_DIR, RESULTS_DIR):
    folder.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

SEED = 42


# ==========================================
# МОДЕЛЬ ЛИНЕЙНОЙ РЕГРЕССИИ
# ==========================================

class LinearRegression(nn.Module):
    """Простая линейная регрессия на PyTorch."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def make_features(X, extended=False):
    """
    Создание новых признаков:
    - квадраты;
    - попарные произведения;
    - среднее;
    - дисперсия.
    """

    X = np.asarray(X, dtype=np.float64)

    if not extended:
        return X.copy()

    columns = [X, X ** 2]

    # Попарные взаимодействия признаков
    columns += [
        (X[:, i] * X[:, j]).reshape(-1, 1)
        for i, j in combinations(range(X.shape[1]), 2)
    ]

    # Среднее значение признаков для каждого объекта
    columns.append(X.mean(axis=1, keepdims=True))

    # Дисперсия признаков для каждого объекта
    columns.append(X.var(axis=1, keepdims=True))

    return np.hstack(columns)


# ==========================================
# ПОДГОТОВКА ДАННЫХ
# ==========================================

def prepare_data():
    """Загрузка, разделение и нормализация Diabetes."""

    dataset = load_diabetes()

    X = dataset.data
    y = dataset.target

    # 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=SEED
    )

    # Нормализация только по тренировочным данным
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Нормализация целевой переменной
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_mean, y_std
    )


# ==========================================
# ОБЩАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ==========================================

def train_and_evaluate(
    X_train,
    y_train,
    X_eval,
    y_eval,
    y_mean,
    y_std,
    optimizer_name="Adam",
    lr=0.01,
    batch_size=32,
    epochs=90
):
    """Обучение модели и расчёт метрик качества."""

    torch.manual_seed(SEED)

    model = LinearRegression(X_train.shape[1])

    # Доступные оптимизаторы
    optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop
    }

    if optimizer_name not in optimizers:
        raise ValueError("Неизвестный оптимизатор")

    optimizer = optimizers[optimizer_name](
        model.parameters(),
        lr=lr
    )

    criterion = nn.MSELoss()

    # Преобразование тренировочных данных в тензоры
    X_tensor = torch.tensor(
        X_train, dtype=torch.float32
    )

    y_tensor = torch.tensor(
        (y_train - y_mean) / y_std,
        dtype=torch.float32
    ).reshape(-1, 1)

    # DataLoader для обучения по батчам
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    history = []

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(batch_X)

        avg_loss = loss_sum / len(loader.dataset)
        history.append(avg_loss)

    # Оценка качества
    model.eval()

    with torch.no_grad():
        X_eval_tensor = torch.tensor(
            X_eval, dtype=torch.float32
        )

        pred_scaled = model(X_eval_tensor)

        # Возвращаем предсказания в исходный масштаб
        predictions = (
            pred_scaled.numpy().ravel() * y_std + y_mean
        )

    mse = mean_squared_error(y_eval, predictions)

    metrics = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_eval, predictions))
    }

    return model, history, metrics


# ==========================================
# СОХРАНЕНИЕ ТАБЛИЦ
# ==========================================

def save_csv(filename, rows, columns):
    """Сохранение результатов экспериментов в CSV."""

    path = RESULTS_DIR / filename

    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# ГРАФИКИ ГИПЕРПАРАМЕТРОВ
# ==========================================

def plot_hyperparameters(rows):
    """Визуализация влияния гиперпараметров на RMSE."""

    settings = [
        ("optimizer", "optimizers.png", "Optimizer"),
        ("lr", "learning_rates.png", "Learning rate"),
        ("batch_size", "batch_sizes.png", "Batch size")
    ]

    for key, filename, xlabel in settings:

        values = list(
            dict.fromkeys(row[key] for row in rows)
        )

        averages = [
            np.mean([
                row["RMSE"]
                for row in rows
                if row[key] == value
            ])
            for value in values
        ]

        plt.figure(figsize=(7, 4))

        positions = np.arange(len(values))

        plt.bar(positions, averages)
        plt.xticks(positions, [str(v) for v in values])

        plt.xlabel(xlabel)
        plt.ylabel("Mean validation RMSE")
        plt.title(f"Effect of {xlabel}")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename)
        plt.close()


# ==========================================
# ЗАДАНИЕ 3.1
# ИССЛЕДОВАНИЕ ГИПЕРПАРАМЕТРОВ
# ==========================================

def hyperparameter_experiments(data):
    """Перебор 27 комбинаций гиперпараметров."""

    (
        X_train, X_val, _,
        y_train, y_val, _,
        y_mean, y_std
    ) = data

    rows = []
    best = None

    # Все комбинации трёх параметров
    combinations_list = product(
        ("SGD", "Adam", "RMSprop"),
        (0.001, 0.01, 0.05),
        (16, 32, 64)
    )

    for optimizer_name, lr, batch_size in combinations_list:

        _, _, metrics = train_and_evaluate(
            X_train,
            y_train,
            X_val,
            y_val,
            y_mean,
            y_std,
            optimizer_name=optimizer_name,
            lr=lr,
            batch_size=batch_size,
            epochs=90
        )

        row = {
            "optimizer": optimizer_name,
            "lr": lr,
            "batch_size": batch_size,
            **metrics
        }

        rows.append(row)

        logging.info(
            "3.1: %s lr=%s batch=%s RMSE=%.2f R2=%.3f",
            optimizer_name,
            lr,
            batch_size,
            metrics["RMSE"],
            metrics["R2"]
        )

        # Поиск лучшей комбинации
        if best is None or row["RMSE"] < best["RMSE"]:
            best = row

    # Сохраняем результаты
    save_csv(
        "hyperparameters.csv",
        rows,
        ["optimizer", "lr", "batch_size",
         "MSE", "RMSE", "R2"]
    )

    plot_hyperparameters(rows)

    logging.info(
        "\nЛучшая комбинация на validation: %s",
        best
    )

    return best


# ==========================================
# ЗАДАНИЕ 3.2
# FEATURE ENGINEERING
# ==========================================

def feature_engineering_experiments(data, best):
    """Сравнение базовой модели с расширенной."""

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_mean, y_std
    ) = data

    results = []

    experiments = [
        ("Baseline", False),
        ("Feature engineering", True)
    ]

    for label, extended in experiments:

        # Формирование признаков
        train_features = make_features(X_train, extended)
        val_features = make_features(X_val, extended)
        test_features = make_features(X_test, extended)

        # Дополнительное масштабирование новых признаков
        scaler = StandardScaler()

        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        # Обучение с выбранными гиперпараметрами
        model, history, val_metrics = train_and_evaluate(
            train_features,
            y_train,
            val_features,
            y_val,
            y_mean,
            y_std,
            optimizer_name=best["optimizer"],
            lr=best["lr"],
            batch_size=best["batch_size"],
            epochs=150
        )

        # Проверяем на тестовой выборке
        model.eval()

        with torch.no_grad():

            X_test_tensor = torch.tensor(
                test_features, dtype=torch.float32
            )

            pred = model(X_test_tensor)

            pred = (
                pred.numpy().ravel() * y_std + y_mean
            )

        mse = mean_squared_error(y_test, pred)

        test_metrics = {
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_test, pred))
        }

        result = {
            "model": label,
            "features": train_features.shape[1],
            "validation_RMSE": val_metrics["RMSE"],
            "test_MSE": test_metrics["MSE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"]
        }

        results.append(result)

        logging.info(
            "3.2: %s, признаков=%d, test RMSE=%.2f, R2=%.3f",
            label,
            result["features"],
            result["test_RMSE"],
            result["test_R2"]
        )

        # Сохраняем модель
        model_name = (
            "extended_model.pth"
            if extended else "baseline_model.pth"
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_mean": scaler.mean_,
                "feature_scale": scaler.scale_,
                "y_mean": y_mean,
                "y_std": y_std
            },
            MODELS_DIR / model_name
        )

        # График обучения
        plt.figure(figsize=(7, 4))
        plt.plot(history)

        plt.xlabel("Epoch")
        plt.ylabel("Training MSE (scaled target)")
        plt.title(label)

        plt.tight_layout()

        filename = (
            "extended_loss.png"
            if extended else "baseline_loss.png"
        )

        plt.savefig(PLOTS_DIR / filename)
        plt.close()

    # Таблица сравнения моделей
    save_csv(
        "feature_comparison.csv",
        results,
        [
            "model",
            "features",
            "validation_RMSE",
            "test_MSE",
            "test_RMSE",
            "test_R2"
        ]
    )

    # График сравнения RMSE
    plt.figure(figsize=(7, 4))

    plt.bar(
        [r["model"] for r in results],
        [r["test_RMSE"] for r in results]
    )

    plt.ylabel("Test RMSE (lower is better)")
    plt.title("Baseline vs Feature Engineering")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_comparison.png")
    plt.close()

    return results


# ==========================================
# ПРОСТЫЕ UNIT-ТЕСТЫ
# ==========================================

def run_tests():
    """Проверка критических функций."""

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    basic = make_features(X)
    extended = make_features(X, extended=True)

    # Проверяем размерности
    assert basic.shape == (2, 2)
    assert extended.shape == (2, 7)

    # Проверяем квадраты
    assert np.allclose(
        extended[:, 2:4], X ** 2
    )

    # Проверяем взаимодействие
    assert np.allclose(
        extended[:, 4],
        X[:, 0] * X[:, 1]
    )

    # Проверяем среднее
    assert np.allclose(
        extended[:, -2],
        X.mean(axis=1)
    )

    # Проверяем дисперсию
    assert np.allclose(
        extended[:, -1],
        X.var(axis=1)
    )

    # Исходные признаки не должны изменяться
    assert np.array_equal(X, basic)

    logging.info("Мини-тесты пройдены")


# ==========================================
# ЗАПУСК ПРОГРАММЫ
# ==========================================

if __name__ == "__main__":

    run_tests()

    data = prepare_data()

    logging.info("=== ЗАДАНИЕ 3.1 ===")

    best = hyperparameter_experiments(data)

    logging.info("\n=== ЗАДАНИЕ 3.2 ===")

    feature_engineering_experiments(data, best)

    logging.info(
        "\nГотово. Графики: plots/, "
        "таблицы: results/, модели: models/"
    )
