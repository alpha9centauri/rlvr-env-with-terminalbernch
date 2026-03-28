import random
from typing import List

from src.env import Task


WORDS = [
    "banana", "apple", "orange", "strawberry", "blueberry",
    "pineapple", "watermelon", "mango", "papaya", "cherry",
    "elephant", "giraffe", "hippopotamus", "rhinoceros", "crocodile",
    "mississippi", "communication", "programming", "mathematics", "philosophy",
]

EXACT_MATCH_TARGETS = [
    "hello", "world", "foo", "bar", "test", "42", "python", "moon",
    "sun", "star", "red", "blue", "green", "cat", "dog",
]


def generate_counting_task(task_id: str) -> Task:
    word = random.choice(WORDS)
    letter = random.choice(list(set(word.lower())))
    count = word.lower().count(letter)
    return Task(
        task_id=task_id,
        instruction=f"Count how many times letter '{letter}' appears in word '{word}'. Return only the final answer.",
        expected_answer=str(count),
        task_type="counting",
    )


def generate_arithmetic_task(task_id: str) -> Task:
    a = random.randint(1, 500)
    b = random.randint(1, 500)
    op = random.choice(["+", "-", "*"])
    if op == "+":
        answer = a + b
        instruction = f"What is {a} + {b}? Return only the final answer."
    elif op == "-":
        if a < b:
            a, b = b, a
        answer = a - b
        instruction = f"What is {a} - {b}? Return only the final answer."
    else:
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        answer = a * b
        instruction = f"What is {a} * {b}? Return only the final answer."
    return Task(
        task_id=task_id,
        instruction=instruction,
        expected_answer=str(answer),
        task_type="arithmetic",
    )


def generate_exact_match_task(task_id: str) -> Task:
    target = random.choice(EXACT_MATCH_TARGETS)
    return Task(
        task_id=task_id,
        instruction=f"Output exactly: {target}",
        expected_answer=target,
        task_type="exact_match",
    )


def generate_task_set(n: int = 20, seed: int = 42) -> List[Task]:
    random.seed(seed)
    tasks = []
    generators = [generate_counting_task, generate_arithmetic_task, generate_exact_match_task]
    for i in range(n):
        gen = generators[i % len(generators)]
        tasks.append(gen(task_id=f"synthetic_{i + 1:03d}"))
    return tasks
