def hIndex(citaions):
    if citaions == [] or citations ==[0]:
        return 0
    citaions.sort()
    citaions.reverse()
    for h,c in enumerate(citaions):
        if h+1 > c:
            return h
    return len(citaions)
