def zero_pad(length, *lists, value=0):
    "Zero-pad the lists up to the specified length"
    for l in lists:
        remaining = length - len(l)
        l += [value] * remaining
