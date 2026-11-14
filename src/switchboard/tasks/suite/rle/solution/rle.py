"""Run-length encoding: "aaab" <-> "3a1b"."""


def encode(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        out.append(f"{j - i}{text[i]}")
        i = j
    return "".join(out)


def decode(code: str) -> str:
    out: list[str] = []
    n = ""
    for ch in code:
        if ch.isdigit():
            n += ch
        else:
            out.append(ch * int(n))
            n = ""
    return "".join(out)
