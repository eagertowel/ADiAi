
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from torch.utils.data import Dataset, DataLoader

from linreg_torch import LinearRegression
from logreg_torch import LogisticRegression


# Настройки
logging.basicConfig(level=logging.INFO, format="%(message)s")

torch.manual_seed(42)
np.random.seed(42)



# Папка, в которой находится текущий Python-файл
BASE_DIR = Path(__file__).resolve().parent

# Создаем папки относительно файла, а не рабочего каталога PyCharm
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

for folder in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# ЗАДАНИЕ 2.1. КАСТОМНЫЙ DATASET
# ==========================================

class CSVDataset(Dataset):
    """
    Собственный Dataset для CSV-файлов.

    Возможности:
    - загрузка CSV;
    - нормализация числовых признаков;
    - кодирование категориальных признаков;
    - обработка пропущенных значений;
    - поддержка регрессии и бинарной классификации.
    """

    def __init__(
        self,
        csv_path,
        target_column,
        task="regression",
        indices=None,
        preprocessor=None,
        label_encoder=None
    ):
        # Загружаем CSV-файл
        df = pd.read_csv(csv_path)

        if target_column not in df.columns:
            raise ValueError("Целевая колонка не найдена")

        if task not in ("regression", "classification"):
            raise ValueError("Неизвестный тип задачи")

        # Выбираем нужные строки
        if indices is not None:
            df = df.iloc[indices].copy()

        # Разделяем признаки и целевую переменную
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Определяем типы признаков
        numeric_columns = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = [
            col for col in X.columns
            if col not in numeric_columns
        ]

        # Обработка числовых признаков
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Обработка категориальных признаков
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        # Объединяем обработку признаков
        if preprocessor is None:
            transformers = []

            if numeric_columns:
                transformers.append(
                    ("num", numeric_transformer, numeric_columns)
                )

            if categorical_columns:
                transformers.append(
                    ("cat", categorical_transformer, categorical_columns)
                )

            if not transformers:
                raise ValueError("В датасете нет признаков")

            self.preprocessor = ColumnTransformer(transformers)

            # Обучаем преобразования на тренировочных данных
            X_processed = self.preprocessor.fit_transform(X)

        else:
            self.preprocessor = preprocessor

            # Для тестовых данных используем готовые преобразования
            X_processed = self.preprocessor.transform(X)

        # Подготовка целевой переменной
        self.label_encoder = label_encoder

        if task == "classification":
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = self.label_encoder.transform(y)

            if len(self.label_encoder.classes_) != 2:
                raise ValueError(
                    "Для бинарной классификации нужно 2 класса"
                )
        else:
            y_processed = pd.to_numeric(
                y, errors="raise"
            ).to_numpy()

        if pd.isna(y_processed).any():
            raise ValueError("Целевая переменная содержит пропуски")

        # Преобразуем данные в тензоры PyTorch
        self.X = torch.tensor(
            np.asarray(X_processed, dtype=np.float32)
        )

        self.y = torch.tensor(
            np.asarray(y_processed, dtype=np.float32)
        ).reshape(-1, 1)

        self.task = task

        logging.info(
            f"Dataset: {len(self.X)} строк, "
            f"{self.X.shape[1]} признаков"
        )

    def __len__(self):
        """Количество объектов в датасете."""
        return len(self.X)

    def __getitem__(self, idx):
        """Возвращает признаки и правильный ответ."""
        return self.X[idx], self.y[idx]


# ==========================================
# ЗАДАНИЕ 2.2. ПОДГОТОВКА CSV-ДАТАСЕТОВ
# ==========================================

def create_csv_files():
    """Сохраняем два встроенных датасета sklearn в CSV."""

    # Датасет для регрессии
    diabetes = load_diabetes()

    df_reg = pd.DataFrame(
        diabetes.data,
        columns=diabetes.feature_names
    )

    df_reg["target"] = diabetes.target

    # Добавим категориальный признак для проверки кодирования
    df_reg["bmi_category"] = np.where(
        df_reg["bmi"] > df_reg["bmi"].median(),
        "high",
        "low"
    )

    df_reg.to_csv("data/diabetes.csv", index=False)

    # Датасет для классификации
    cancer = load_breast_cancer()

    df_clf = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    df_clf["target"] = cancer.target

    df_clf["radius_category"] = np.where(
        df_clf["mean radius"] > df_clf["mean radius"].median(),
        "large",
        "small"
    )

    df_clf.to_csv("data/breast_cancer.csv", index=False)

    logging.info("CSV-файлы созданы в папке data/")


def prepare_datasets(csv_path, task):
    """Создание обучающего и тестового датасетов."""

    df = pd.read_csv(csv_path)

    indices = np.arange(len(df))

    stratify = df["target"] if task == "classification" else None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    train_dataset = CSVDataset(
        csv_path,
        target_column="target",
        task=task,
        indices=train_idx
    )

    test_dataset = CSVDataset(
        csv_path,
        target_column="target",
        task=task,
        indices=test_idx,
        preprocessor=train_dataset.preprocessor,
        label_encoder=train_dataset.label_encoder
    )

    return train_dataset, test_dataset


# ==========================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# ==========================================

def train_model(
    model,
    train_loader,
    task,
    epochs=200,
    lr=0.01
):
    """Общий цикл обучения для двух типов регрессии."""

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if task == "regression":
        criterion = nn.MSELoss()

        # Нормализуем целевые значения для обучения
        all_y = train_loader.dataset.y
        y_mean = all_y.mean()
        y_std = all_y.std().clamp(min=1e-8)

    else:
        criterion = nn.BCEWithLogitsLoss()
        y_mean = None
        y_std = None

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)

            if task == "regression":
                target = (y_batch - y_mean) / y_std
            else:
                target = y_batch

            loss = criterion(predictions, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(train_loader.dataset)
        history.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            logging.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f}"
            )

    return history, y_mean, y_std


# ==========================================
# ЭКСПЕРИМЕНТ 1. ЛИНЕЙНАЯ РЕГРЕССИЯ
# ==========================================

def regression_experiment():
    """Обучение линейной регрессии на Diabetes."""

    print("\n=== ЛИНЕЙНАЯ РЕГРЕССИЯ ===", flush=True)

    train_dataset, test_dataset = prepare_datasets(
        "data/diabetes.csv",
        task="regression"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = LinearRegression(
        in_features=train_dataset.X.shape[1]
    )

    history, y_mean, y_std = train_model(
        model,
        train_loader,
        task="regression",
        epochs=200,
        lr=0.01
    )

    model.eval()

    with torch.no_grad():
        predictions = model(test_dataset.X)

        # Возвращаем предсказания в исходный масштаб
        predictions = predictions * y_std + y_mean

    y_true = test_dataset.y.numpy()
    y_pred = predictions.numpy()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    torch.save(
        model.state_dict(),
        MODELS_DIR / "diabetes_model.pth"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title("Linear Regression - Diabetes")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE Loss")
    plt.grid(True)
    plt.savefig("plots/diabetes_loss.png")
    plt.close()


# ==========================================
# ЭКСПЕРИМЕНТ 2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ==========================================

def classification_experiment():
    """Обучение бинарной классификации на Breast Cancer."""

    print("\n=== ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ===", flush=True)

    train_dataset, test_dataset = prepare_datasets(
        "data/breast_cancer.csv",
        task="classification"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    model = LogisticRegression(
        in_features=train_dataset.X.shape[1]
    )

    history, _, _ = train_model(
        model,
        train_loader,
        task="classification",
        epochs=200,
        lr=0.01
    )

    model.eval()

    with torch.no_grad():
        logits = model(test_dataset.X)

        # Превращаем логиты в вероятности
        probabilities = torch.sigmoid(logits)

        # Определяем класс по порогу 0.5
        predictions = (probabilities >= 0.5).int()

    y_true = test_dataset.y.numpy().ravel().astype(int)
    y_pred = predictions.numpy().ravel()
    y_prob = probabilities.numpy().ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    torch.save(
        model.state_dict(),
        "models/breast_cancer_model.pth"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title("Logistic Regression - Breast Cancer")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(True)
    plt.savefig("plots/breast_cancer_loss.png")
    plt.close()


# ==========================================
# ПРОВЕРКА DATASET
# ==========================================

def test_csv_dataset():
    """Простые проверки корректности Dataset."""

    train_dataset, test_dataset = prepare_datasets(
        "data/diabetes.csv",
        task="regression"
    )

    assert len(train_dataset) > 0
    assert len(test_dataset) > 0

    X, y = train_dataset[0]

    assert X.ndim == 1
    assert y.shape == (1,)
    assert torch.isfinite(X).all()

    # Обработка должна давать одинаковое число признаков
    assert train_dataset.X.shape[1] == test_dataset.X.shape[1]

    logging.info("Проверки CSVDataset пройдены")


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    create_csv_files()

    print("\n=== ЗАДАНИЕ 2.1 ===", flush=True)
    test_csv_dataset()

    print("\n=== ЗАДАНИЕ 2.2 ===", flush=True)

    regression_experiment()
    classification_experiment()
