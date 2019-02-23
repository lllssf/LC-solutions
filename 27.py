# -*- coding: utf-8 -*-
def removeElement(nums,val):
    #ERROR:修改遍历的列表
    for x in nums[:]:
        if x == val:
            nums.remove(x)
    return nums
print(removeElement([0,1,2,2,3,0,4,2,],2))
