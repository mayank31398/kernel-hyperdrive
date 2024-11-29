def ceil_divide(x: int, y: int) -> int:
    return (x + y - 1) // y


def check_power_of_2(n: int) -> bool:
    return n & (n - 1) == 0 and n != 0


def get_block_sizes_powers_of_2(start: int, end: int) -> list[int]:
    assert check_power_of_2(start), "start is not a power of 2"
    assert check_power_of_2(end), "end is not a power of 2"

    output = []
    n = start
    while n <= end:
        output.append(n)
        n = n << 1

    return output
