def increasingTriple(nums):
    n = len(nums)
    if n < 3: return False
    i,j,k = 0,1,2
    while k < n:
        if nums[i] < nums[j] and nums[j]<nums[k]:
            return True
        while nums[i] >= nums[j]:
            i = j
            j = i + 1
            k = i + 2
            if k > n-1:
                return False
        while nums[j] >= nums[k]:
            k += 1
            if k > n -1:
                j,k = j+1, j+2
                break
    return False

print(increasingTriple([5,1,5,5,2,4]))
def increasingTriplet1(nums):
    first = second = float('inf')
    for n in nums:
        if n <= first:
            first = n
        elif n <= second:
            second = n
        else:
            return True
    return False
