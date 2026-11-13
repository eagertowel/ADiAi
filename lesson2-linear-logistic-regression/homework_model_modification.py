
import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ==========================================
# ЗАДАНИЕ 1.1. ЛИНЕЙНАЯ РЕГРЕССИЯ
# ==========================================

class LinearRegression(nn.Module):
    """Линейная регрессия с поддержкой L1 и L2."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_regression(
    model,
    train_loader,
    val_loader,
    lr=0.01,
    epochs=500,
    l1=0.001,
    l2=0.001,
    patience=20
):
    """Обучение с регуляризацией и Early Stopping."""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)

            # Основная ошибка модели
            mse_loss = criterion(predictions, y_batch)

            # L1 и L2 регуляризация весов
            weights = model.linear.weight

            l1_loss = l1 * torch.sum(torch.abs(weights))
            l2_loss = l2 * torch.sum(weights ** 2)

            loss = mse_loss + l1_loss + l2_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Проверка качества на валидационных данных
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                total_val_loss += loss.item() * len(X_batch)

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 20 == 0:
            logging.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}"
            )

        # Early Stopping
        if avg_val_loss < best_loss - 1e-4:
            best_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            logging.info(
                f"Early Stopping на эпохе {epoch + 1}"
            )
            break

    # Возвращаем лучшие веса модели
    model.load_state_dict(best_weights)

    return train_losses, val_losses


def run_linear_regression():
    """Запуск линейной регрессии на Diabetes."""

    data = load_diabetes()
    X = data.data
    y = data.target.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=32
    )

    model = LinearRegression(in_features=X_train.shape[1])

    train_losses, val_losses = train_linear_regression(
        model,
        train_loader,
        val_loader,
        lr=0.01,
        epochs=500,
        l1=0.001,
        l2=0.001,
        patience=20
    )

    torch.save(
        model.state_dict(),
        "models/linear_regression.pth"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Linear Regression Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/linear_regression.png")
    plt.close()

    logging.info(
        f"Лучшая validation MSE: {min(val_losses):.4f}"
    )


# ==========================================
# ЗАДАНИЕ 1.2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ==========================================

class LogisticRegression(nn.Module):
    """Логистическая регрессия для нескольких классов."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_logistic_regression(
    model,
    train_loader,
    lr=0.01,
    epochs=200
):
    """Обучение многоклассовой логистической регрессии."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            logging.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.4f}"
            )

    return losses


def evaluate_logistic_regression(model, X_test, y_test):
    """Расчёт метрик и построение Confusion Matrix."""

    model.eval()

    with torch.no_grad():
        logits = model(X_test)

        # Вероятности принадлежности к классам
        probabilities = torch.softmax(logits, dim=1)

        # Класс с максимальной вероятностью
        predictions = torch.argmax(probabilities, dim=1)

    y_true = y_test.numpy()
    y_pred = predictions.numpy()
    y_prob = probabilities.numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="weighted",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    print("\nРезультаты классификации:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Setosa", "Versicolor", "Virginica"]
    )

    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()


def run_logistic_regression():
    """Запуск классификации на Iris."""

    data = load_iris()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        dataset, batch_size=16, shuffle=True
    )

    model = LogisticRegression(
        in_features=X_train.shape[1],
        num_classes=3
    )

    losses = train_logistic_regression(
        model,
        train_loader,
        lr=0.01,
        epochs=200
    )

    torch.save(
        model.state_dict(),
        "models/logistic_regression.pth"
    )

    evaluate_logistic_regression(
        model, X_test, y_test
    )

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Regression Training")
    plt.grid(True)
    plt.savefig("plots/logistic_regression.png")
    plt.close()


# ==========================================
# ЗАПУСК ПРОГРАММЫ
# ==========================================

if __name__ == "__main__":

    print("=== ЗАДАНИЕ 1.1 ===")
    run_linear_regression()

    print("\n=== ЗАДАНИЕ 1.2 ===")
    run_logistic_regression()
