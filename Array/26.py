def removeDuplicates(nums):
    dup_nums = []
    for x in nums:
        if x not in dup_nums:
            dup_nums.append(x)
    nums = dup_nums
    return nums
print(removeDuplicates([0,0,1,1,1,2,2,3,3,4]))

def removeDuplicates_1(nums):
    dupnums = [x for x in nums]
    i = 0
    for x in dupnums[:]:
        if x in dupnums[i+1:]:
            nums.remove(x)
        i = i+1
        if i == len(dupnums)-1:
            return nums

print(removeDuplicates_1([0,0,1,1,1,2,2,3,3,4]))
