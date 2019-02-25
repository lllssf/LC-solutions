def hIndex_bi(citations):
    xitations.reverse()
    low, high = 1, len(citations)
    while low <= high:
        mid = (low + high) / 2
        if mid == citations[mid-1]:
            return mid
        elif mid < citations[mid-1]:
            low = mid + 1
        else:
            high = mid - 1
    return low-1
