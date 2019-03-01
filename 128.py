def longestConsecutive(nums):
    if not nums: return 0
    length,length_max = 0,[]
    nums.sort()
    start = nums[0]
    for n in nums[1:]:
        if n == start + 1:
            length += 1
            start += 1
        elif n > start+1:
            length_max.append(length)
            start = n
            length = 1
    length_max.append(length)
    return max(length_max)
