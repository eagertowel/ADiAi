"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set('aeiouAEIOU')
    return sum(1 for char in text if char in vowels)
#tests 1



def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))
#tests 2

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')
#tests 3

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps=0
    while number>=10:
        prod=1
        for digit in str(number):
            prod*= int(digit)
        number=prod
        steps+=1
    return steps
#tests 4

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    total_squared_error = sum((p-e)**2 for p,e in zip(predicted, expected))
    return total_squared_error/len(predicted)
#tests 5

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result=[]
    count=0
    while number%2 ==0:
        count+=1
        number//=2
    if count>0:
        result.append(f'({2**count})' if count>1 else '(2)')

    d=3
    while d*d <=number:
        if number%d==0:
            count=0
            while number%d==0:
                count +=1
                number//=d
            result.append(f'({d}**{count})' if count>1 else f'{d}')
        d+=2
    if number>1:
        result.append(f'({number})')
    return ''.join(result)
#tests 6


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k=1
    total_cubes=0
    while total_cubes<cube_count:
        total_cubes+=k**2
        if total_cubes==cube_count:
            return k
        k+=1
    return 'It is impossible'
#tests 7

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    s=str(data.value)
    length=len(s)
    if length<=2:
        return True

    half=(length-1)//2
    left_sum=sum(int(digit) for digit in s[:half])
    right_sum=sum(int(digit) for digit in s[-half:])

    return left_sum==right_sum

#tests 8


class PositiveIntegerInput:
    def __init__(self, value: int):
        self.value = value


def test_is_balanced_number():
    # 1. Пример с нечетным количеством цифр (1+2+3 == 0+0+6)
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True

    # 2. Пример с четным количеством цифр (1+2 != 5+6)
    assert is_balanced_number(PositiveIntegerInput(123456)) is False

    # 3. Однозначное число
    assert is_balanced_number(PositiveIntegerInput(7)) is True

    # 4. Двузначное число (обе суммы 0)
    assert is_balanced_number(PositiveIntegerInput(99)) is True

    # 5. Трехзначное сбалансированное число (1 == 1)
    assert is_balanced_number(PositiveIntegerInput(151)) is True

    # 6. Четырехзначное сбалансированное число (2 == 2)
    assert is_balanced_number(PositiveIntegerInput(2892)) is True

    print("Все тесты успешно пройдены!")


test_is_balanced_number()