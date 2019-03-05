def maximumGap(nums):
    if len(nums) < 2: return 0
    nums.sort()
    gap,maxGap,prev = 0,[],0
    for n in nums:
        if n-prev > gap:
            gap = n-prev
            maxGap.append(gap)
            prev = n
        else: prev = n
    maxGap.append(gap)
    return max(maxGap)

print(maximumGap([3,6,9,1]))
