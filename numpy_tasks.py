"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices_arr = np.array(data.matrices)
    vectors_arr = np.array(data.vectors)

    return np.sum(matrices_arr @ vectors_arr, axis=0)

def binarize(data: BinarizeInput) -> np.ndarray:
    return np.where(data.matrix > data.threshold, 1, 0)

def unique_rows(data: MatrixInput) -> list[list[float]]:
    return [np.unique(row).tolist() for row in data.matrix]

def unique_columns(data: MatrixInput) -> list[list[float]]:
    return [np.unique(col).tolist() for col in data.matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)

    matrix = rng.normal(loc=data.mean, scale=data.std, size=(data.rows, data.columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means = np.mean(matrix, axis=1),
        col_means = np.mean(matrix, axis=0),
        row_vars = np.var(matrix, axis=1),
        col_vars = np.var(matrix, axis=0),
    )

def chess(data: ChessInput) -> np.ndarray:
    matrix = np.full((data.rows, data.columns), data.first)

    matrix[1::2, ::2] = data.second
    matrix[::2, 1::2] = data.second

    return matrix

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image = np.full((data.image_height, data.image_width, 3), data.background_color, dtype=np.uint8)

    cy, cx = data.image_height / 2, data.image_width / 2

    y0 = max(0, int(round(cy - data.height / 2)))
    y1 = min(data.image_height, int(round(cy + data.height / 2)))
    x0 = max(0, int(round(cx - data.width / 2)))
    x1 = min(data.image_width, int(round(cx + data.width / 2)))

    image[y0:y1, x0:x1] = data.shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    image = np.full((data.image_height, data.image_width, 3), data.background_color, dtype=np.uint8)

    cy, cx = (data.image_height - 1) / 2.0, (data.image_width - 1) / 2.0

    y, x = np.ogrid[:data.image_height, :data.image_width]

    mask = ((x - cx) ** 2 / data.semi_axis_x ** 2 + (y - cy) ** 2 / data.semi_axis_y ** 2) <= 1.0

    image[mask] = data.shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    v = data.values

    mean_val, var_val, std_val = float(np.mean(v)), float(np.var(v)), float(np.std(v))

    if len(v) > 2:
        maxima = np.where((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
        minima = np.where((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
    else:
        maxima, minima = np.array([], dtype=int), np.array([], dtype=int)

    ma = np.convolve(v, np.ones(data.window) / data.window, mode="valid")

    return TimeSeriesStatistics(
        mean = mean_val, variance = var_val, std = std_val,
        local_maxima = maxima, local_minima = minima, moving_average = ma
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    c = data.class_count
    if c is None:
        c = int(np.max(data.labels)) + 1 if len(data.labels) > 0 else 0

    return np.eye(c, dtype=int)[data.labels]