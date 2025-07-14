from lib import create_circular_queue


def simulate_card_game(n: int) -> int:
    """
    카드2 문제의 시뮬레이션
    맨 위 카드를 버리고, 그 다음 카드를 맨 아래로 이동
    """

    q = create_circular_queue(n)
    while len(q) > 1:
        q.popleft()
        q.append(q.popleft())

    return q.popleft()


def solve_card2() -> None:
    """입, 출력 format"""
    n: int = int(input())
    result: int = simulate_card_game(n)
    print(result)


if __name__ == "__main__":
    solve_card2()
