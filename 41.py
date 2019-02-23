# -*- coding: utf-8 -*-
def firstMissingPositive(nums):
    #先考虑特殊情况
    if nums == []:
        return 1
    if len(nums) == 1:
        if nums[0] <1 or nums[0] >1:
            return 1
        else:
            return 2
    nums.sort()
    for x in nums[:]:
        if x < 1:
            nums.remove(x)
    if nums == [] or nums[0]>1:
        return 1
    for y in nums[:]:
        if y+1 not in nums:
            return y+1

print(firstMissingPositive([0,1,2,9,8,-1,-2,7,6,4,5,3,223423]))

# -*- coding: utf-8 -*-
def firstMissingPositive1(nums):
    #先考虑特殊情况
    if nums == []:
        return 1
    if len(nums) == 1:
        if nums[0] <1 or nums[0] >1:
            return 1
        else:
            return 2
    nums.sort()
    for x in nums[:]:
        if x < 1:
            nums.remove(x)
    if nums == [] or nums[0]>1:
        return 1
    for i in range(len(nums)-1):
        if nums[i] + 1 !=nums[i+1]:
            return nums[i]+1
    return nums[-1]+1

print(firstMissingPositive1([0,1,2,9,8,-1,-2,7,6,4,5,3,223423]))
