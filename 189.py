def rotate(nums,k):
    for i in range(k):  #number of step
        nums.insert(0,nums[-1])
        nums.pop(-1)

    return nums

print(rotate([1,2,3,4,5,6,7],3))

def rotate_1(nums,k):
    nums.reverse()
    nums[:k] = nums[k-1::-1]
    nums[k:] = nums[:k-1:-1]
    return nums

print(rotate_1([1,2,3,4,5,6,7],3))

def rotate_2(nums,k):
    nums.reverse()
    nums[:k] = list(reversed(nums[:k]))
    #reversed()返回的是一个iterator，可以反转tuple,string,list,range
    nums[k:] = list(reversed(nums[k:]))
    return nums

print(rotate_2([1,2,3,4,5,6,7],3))
