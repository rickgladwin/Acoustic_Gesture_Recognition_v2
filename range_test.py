# test range sizes

test_range: list[float] = [ (x / 100) for x in range(200, 610, 25) ]

print(f"test_range size: {len(test_range)}")
print(f"{test_range=}")
