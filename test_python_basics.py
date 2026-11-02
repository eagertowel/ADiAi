"""Небольшие pytest-тесты к python_basics.py (по требованиям README)."""
from types import SimpleNamespace
import pytest
from python_basics import (
    count_vowels, has_unique_characters, count_one_bits,
    multiplicative_persistence, mse, prime_factorization,
    pyramid, is_balanced_number,
)


def inp(value):
    """Объект с полем value, которое используют функции."""
    return SimpleNamespace(value=value)


@pytest.mark.parametrize('s,expected', [('Hello WORLD', 3), ('bcdf', 0), ('', 0), ('AEIOU', 5)])
def test_count_vowels(s, expected):
    assert count_vowels(inp(s)) == expected


@pytest.mark.parametrize('s,expected', [('abc', True), ('hello', False), ('', True), ('Aa', True)])
def test_has_unique_characters(s, expected):
    assert has_unique_characters(inp(s)) is expected


@pytest.mark.parametrize('n,expected', [(1, 1), (7, 3), (8, 1), (10, 2)])
def test_count_one_bits(n, expected):
    assert count_one_bits(inp(n)) == expected


@pytest.mark.parametrize('n,expected', [(4, 0), (39, 3), (999, 4), (10, 1)])
def test_multiplicative_persistence(n, expected):
    assert multiplicative_persistence(inp(n)) == expected


@pytest.mark.parametrize('pred,actual,expected', [
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([1, 3], [3, 1], 4.0),
    ([0, 4, 2], [1, 1, 3], 11 / 3),
])
def test_mse(pred, actual, expected):
    data = SimpleNamespace(predicted=pred, expected=actual)
    assert mse(data) == pytest.approx(expected)


@pytest.mark.parametrize('n,expected', [
    (1, ''),
    (2, '(2)'),
    (13, '(13)'),
    (12, '(2**2)(3)'),
    (86240, '(2**5)(5)(7**2)(11)'),
])
def test_prime_factorization(n, expected):
    # Ожидаемый формат задан в README. Тест обнаруживает несоответствие реализации.
    assert prime_factorization(inp(n)) == expected


@pytest.mark.parametrize('n,expected', [
    (1, 1), (5, 2), (14, 3), (2, 'It is impossible'), (15, 'It is impossible'),
])
def test_pyramid(n, expected):
    assert pyramid(inp(n)) == expected


@pytest.mark.parametrize('n,expected', [
    (7, True), (99, True), (151, True),
    (1234006, True), (123456, False),
    (1941, True),   # 1 == 1, две средние цифры не участвуют
    (1234, False),  # 1 != 4
])
def test_is_balanced_number(n, expected):
    assert is_balanced_number(inp(n)) is expected
