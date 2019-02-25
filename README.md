# Array
## Primary
### #27

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

### #26

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
### #80

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
### #189

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

### #41

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

### #299
这道题开始忘了用dict，在计算cow的时候很笨的用了两层循环

```
def getHint(secret,guess):
    bull = 0
    cow = 0
    index = list(range(len(secret)))
    # calculate the value of bulls and get rid of its index
    for i in index[:]:
        if secret[i] == guess[i]:
            bull += 1
            index.remove(i)
    # get the value of cows
    for i in index[:]:
        for j in range(len(guess)):
            if secret[i] == guess[j] and j in index: # to avoid being counted before
                cow += 1
                index.remove(j)
                break
    return '%sA%sB' %(bull,cow)

print(getHint('1123','0111'))
```
运行速度很慢
然后在高票答案里发现了一个很简单的解法    

```
from collections import Counter
def getHint1(secret,guess):
    '''
    use Counter to count guess and secret and sum their overlap.
    use zip to counter bulls
    '''
    s,g=Counter(secret),Counter(guess) #return a dict
    bull = sum(i == j for i,j in zip(secret,guess))
    cow = sum((s & g).values()) - bull
    return '%sA%sB' %(bull,cow)

print(getHint1('1123','0111'))
```
与这个解法思路类似用dict的话：

```
def getHint2(secret,guess):
    bull = 0
    cow = 0
    counts = {} #caiculate the counts of s
    for i,s in enumerate(secret):
        if s == guess[i]:
            bull += 1
        else:
            counts[s] = counts.get(s,0) + 1
    for i,s in enumerate(secret):
        if guess[i]!= s and counts.get(guess[i],0)!=0:
            cow += 1
            counts[guess[i]] -= 1
    return '%sA%sB' %(bull,cow)

print(getHint2('1123','0111'))
```
后两种算法的计算速度和内存占用量都差不多

### #134

```
def canCompleteCircuit(gas,cost):
    remain = list(map(lambda x:x[0]-x[1],zip(gas,cost)))
    if sum(remian)<0:
        return -1
    accumulate,start =0,0
    for i in range(remain):
        accumulate += reamin[i]
        if accumulate < 0:
            accumulate,start = 0,i+1
    return start
```
内存占用更少的解法：

```
def canComplateCircuit1(gas,cost):
    start,overall,accumulate = 0,0,0
    for i in range(gas):
        accumulate += gas[i]-cost[i]
        overall += gas[i]-cost[i]
        if accumulater < 0:
            start,accumulate = i+1,0
    return start if overall>0 else -1
```
### #274
关于h指数的定义，先看了百度百科：
> 被引次数从高到低排列，直到谋篇论文的序号大于该论文被引次数，那个序号减去1就是h

按照这个思路：

```
def hIndex(citaions):
    if citaions == [] or citations ==[0]:
        return 0
    citaions.sort()
    citaions.reverse()
    for h,c in enumerate(citaions):
        if h+1 > c:
            return h
    return len(citaions)
```
测试时间16ms，内存占用10.8MB

### #275
与上一题一样，只是用了二分查找

```
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
```


### #217
开始我用的前面学到的Counter模块

```
from collections import Counter
def containsDupliacte(nums):
    n = Counter(nums)
    for num in nums:
        if n[num] > 1:
            return True
    return False
```
然后又看到一个很棒的解答：

```
def containDuplicate1(nums):
    return len(set(nums)) != len(nums)
```
> set()创建一个无序不重复元素集

### #55
> **广度优先搜索BFS**(breadth-first search) 适用于解决“**最短路径问题**”(shortest-path problem)。解决这类问题：
>1. 使用图来建立问题模型
>2. 使用广度优先搜索解决问题  

>广度优先搜索算法可帮助回答两类问题：
>1. 从节点A出发，有前往节点B的路径吗？
>2. 从节点A出发，前往节点B哪条路径最短？

这道题就是路径存在与否问题，可以依据当前最远到达距离与下一节点标号作对比。我一开始想得太过复杂了。

```
def canJump(nums):
    far = 0
    for i, step in enumerate(nums):
        if i > far:
            return False
        far = max(far,i+step)
    return True
```
### #45
这道题就是最短路径问题，就用到了BFS，将当前能到达的点作为第一维度的图，由第一维度才能到达的为第二维度等等以此划分，最少节点到达的路径即为最短路径。

```
def jump(nums):
    node_start,node_end,farest,steps = 0,0,0,0
    while node_end < len(nums)-1:
        steps += 1
        for i in range(node_start,node_end+1):
            if i+nums[i] > len(nums)-1:
                return steps
            farest = max(farest,i+num[i])
        node_end,node_start = farest,node_end+1
    return steps
```
