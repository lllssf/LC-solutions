# Leetcode solution

## #27

```
# -*- coding: utf-8 -*-
def removeElement(nums,val):
    #ERROR:修改遍历的列表
    for x in nums[:]:
        if x == val:
            nums.remove(x)
    return nums
print(removeElement([0,1,2,2,3,0,4,2,],2))
```
开始出现的问题就是最开始for循环写成

```
for x in nums:
```
> 我们通常尽量避免修改一个正在进行遍历的列表，避免这种问题的方法是使用**切片操作克隆这个list**

## #26

```
def removeDuplicates(nums):
    dup_nums = []
    for x in nums:
        if x not in dup_nums:
            dup_nums.append(x)
    nums = dup_nums
    return(nums)
print(removeDuplicates([0,0,1,1,1,2,2,3,3,4]))
```
这个初始版本在pycharm上可以准确输出但在LeetCode上nums是不变的，可能是因为不能改变nums的内存位置

改为如下版本

```
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
```
其中出现的问题是开始对nums复制时的命令写为

```
dupnums = nums
```
>这样不能复制nums，只能为现有list引入一个新的名称。

我用了列表推导式新建了一个与nums一样的list，也可以用copy库，但我发现这样的时间少一些，而且占用内存少。
## #80

因为都是有序数列所以和上一题思路一样
```
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


```
## #189

```
def rotate(nums,k):
    for i in range(k):  #number of step
        nums.insert(0,nums[-1])
        nums.pop(-1)

    return nums

print(rotate([1,2,3,4,5,6,7],3))
```
不用循环的方法更快，测试结果超过100%其他提交者，内存也用的不多：

```
def rotate_1(nums,k):
    nums.reverse()
    nums[:k] = nums[k-1::-1]
    nums[k:] = nums[:k-1:-1]
    return nums

print(rotate_1([1,2,3,4,5,6,7],3))
```
要注意的是，reverse()不能反转部分数组，要想反转部分一个是如上使用切片，不过要注意索引标号，end别忘了要多一个。
还有可以

```
def rotate_2(nums,k):
    nums.reverse()
    nums[:k] = list(reversed(nums[:k]))
    nums[k:] = list(reversed(nums[k:]))
    return nums

print(rotate_2([1,2,3,4,5,6,7],3))
```
reversed()返回的是一个iterator，可以反转tuple,string,list,range

## #41

```
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
```
这道题前几次报错都是因为没有考虑特殊情况，运行时间20ms，因为“not in”的算法复杂度好像不是o(n)，所以又改了一下：

```
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
```
这个在LeetCode测试是16ms，快于100%






