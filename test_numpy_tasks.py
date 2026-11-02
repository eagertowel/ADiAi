"""Небольшие pytest-тесты к numpy_tasks.py (включая граничные случаи)."""
from types import SimpleNamespace
import numpy as np
import pytest
from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns, matrix_statistics,
    chess, draw_rectangle, draw_ellipse, analyze_time_series, one_hot,
)


def test_sum_prod():
    a = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 3]]])
    v = np.array([[[1], [2]], [[3], [4]]])
    result = sum_prod(SimpleNamespace(matrices=a, vectors=v))
    np.testing.assert_array_equal(result, [[7], [14]])
    assert result.shape == (2, 1)


def test_binarize():
    source = np.array([[2, 3], [4, -1]])
    result = binarize(SimpleNamespace(matrix=source, threshold=3))
    np.testing.assert_array_equal(result, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(source, [[2, 3], [4, -1]])


def test_unique_rows():
    data = SimpleNamespace(matrix=np.array([[3, 1, 3], [2, 2, 1]]))
    assert unique_rows(data) == [[1, 3], [1, 2]]


def test_unique_columns():
    data = SimpleNamespace(matrix=np.array([[3, 1, 3], [2, 1, 3]]))
    assert unique_columns(data) == [[2, 3], [1], [3]]


def test_matrix_statistics():
    data = SimpleNamespace(rows=2, columns=3, mean=5, std=2, seed=42)
    result = matrix_statistics(data)
    assert result.matrix.shape == (2, 3)
    np.testing.assert_allclose(result.row_means, result.matrix.mean(axis=1))
    np.testing.assert_allclose(result.col_means, result.matrix.mean(axis=0))
    np.testing.assert_allclose(result.row_vars, result.matrix.var(axis=1))
    np.testing.assert_allclose(result.col_vars, result.matrix.var(axis=0))
    np.testing.assert_array_equal(result.matrix, matrix_statistics(data).matrix)


def test_chess():
    data = SimpleNamespace(rows=3, columns=4, first=0, second=1)
    np.testing.assert_array_equal(chess(data), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert chess(SimpleNamespace(rows=1, columns=1, first=5, second=9)).item() == 5


def test_draw_rectangle():
    data = SimpleNamespace(image_height=5, image_width=5, height=3, width=3,
                           background_color=(0, 0, 0), shape_color=(255, 0, 0))
    img = draw_rectangle(data)
    assert img.shape == (5, 5, 3)
    np.testing.assert_array_equal(img[2, 2], [255, 0, 0])
    np.testing.assert_array_equal(img[0, 0], [0, 0, 0])


def test_draw_ellipse():
    data = SimpleNamespace(image_height=5, image_width=5, semi_axis_x=1, semi_axis_y=1,
                           background_color=(0, 0, 0), shape_color=(0, 255, 0))
    img = draw_ellipse(data)
    assert img.shape == (5, 5, 3)
    np.testing.assert_array_equal(img[2, 2], [0, 255, 0])
    np.testing.assert_array_equal(img[0, 0], [0, 0, 0])


def test_analyze_time_series():
    data = SimpleNamespace(values=np.array([1, 3, 1, 2, 0]), window=3)
    result = analyze_time_series(data)
    assert result.mean == pytest.approx(1.4)
    assert result.variance == pytest.approx(np.var(data.values))
    assert result.std == pytest.approx(np.std(data.values))
    np.testing.assert_array_equal(result.local_maxima, [1, 3])
    np.testing.assert_array_equal(result.local_minima, [2])
    np.testing.assert_allclose(result.moving_average, [5 / 3, 2, 1])
    short = analyze_time_series(SimpleNamespace(values=np.array([2, 2]), window=1))
    assert short.local_maxima.size == short.local_minima.size == 0


def test_one_hot():
    expected = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    np.testing.assert_array_equal(one_hot(SimpleNamespace(labels=np.array([0, 2, 3]), class_count=None)), expected)
    explicit = one_hot(SimpleNamespace(labels=np.array([1]), class_count=4))
    np.testing.assert_array_equal(explicit, [[0, 1, 0, 0]])
    assert one_hot(SimpleNamespace(labels=np.array([], dtype=int), class_count=None)).shape == (0, 0)
