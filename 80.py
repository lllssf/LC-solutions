def removeDuplicates(nums):
    dupnums = [x for x in nums]
    i = 0
    for x in dupnums[:]:
        if x in dupnums[i+2:]:
            nums.remove(x)
        i = i+1
        if i == len(dupnums)-2:
            return nums

print(removeDuplicates([0,0,1,1,1,2,2,3,3,3]))
