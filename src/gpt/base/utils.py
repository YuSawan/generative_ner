import re


def regex(txt: str, pattern: str) -> list[tuple[int, int]]:
    recompiler =re.compile(pattern)
    find_expressions = []
    for res in recompiler.finditer(txt):
        find_expressions.append((res.start(), res.end()))
    return find_expressions


def remove_overlapped_offsets(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    removed_offsets = []
    offsets = list(set(offsets))
    for (start, end) in offsets:
        flag = True
        for (s_start, s_end) in offsets:
            if start == s_start and end == s_end:
                continue
            if start >= s_start and end <= s_end:
                flag = False
                break
        if flag:
            removed_offsets.append((start, end))
    return removed_offsets
