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
与上一题一样，只是用了**二分查找**

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
### #121

```
def maxprofit(prices):
    buy,profit = float('inf'),0
    for p in prices:
        buy = min(buy,p)
        profit = max(profit, p-buy)
    return profit
```
### #122

```
def maxprofit(prices):
    buy,profit,date_sell = float('inf'),0,-1
    for i,p in enumerate(prices):
        if i > sell:
            buy = min(buy,p)
            if p > buy:
                date_sell = i
                profit += p-buy
                buy = p
    return profit
```
还有比较简单的写法：

```
def maxProfit(prices):
    return sum(x[0]-x[1] for x in zip(prices[1:],prices[:-1]) if x[0]>x[1] )
```
### #123
开始写得很复杂，后来看到了用**前后遍历**的算法的简单解法，豁然开朗：

```
def maxProfit(self,prices):
    n = len(prices)
    if n <= 1:
        return 0
    forward_profit,backward_profit = [0]*n,[0]*n
    buy = prices[0]
    for i in range(1,n):
        buy = min(buy,prices[i])
        forward_profit[i] = max(forward_profit[i-1],prices[i]-buy)

    sell = prices[-1]
    for i in range(n-2,-1,-1):
        sell = max(sell,prices[i])
        backward_profit[i] = max(backward_profit[i+1],sell-prices[i])

    maxprofit = 0
    for i in range(n):
        maxprofit = max(maxprofit,forward_profit[i]+backward_profit[i])

    return maxprofit
```
### #188
>**动态规划DP**(Dynamic programming),是一种将问题分成子问题，先解决子问题的方法。

这道题我看了各种解法，都不甚明白，后来在discuss里选了速度最快，占用内存最少的一种仔细研究，明白些许。hold[k]是k-1次操作后持有的股票收益减去买股票的金额，profit[k]是完成k次交易后得到的收益。

```
def maxProfit(k,prices):
    n = len(prices)
    if k >= n/2:
        return sum(x-y for x,y in zip(prices[1:],prices[:-1]) if x>y)

    profit, hold = [0]*(k+1),[float('-inf')]*(k+1)
    for p in prices:
        for i in range(1,k+1):
            profit[i] = max(profit[i], hold[i]+p)
            hold[i] = max(hold[i], profit[i-1]-p)
    return profit[k]
```

### #309
按着上一题的思路，开始的解法如下：

```
def maxprofit(prices):
    n = len(prices)
    if n < 2:
        return 0
    buy,profit = [float('-inf')]*n,[0]*n
    buy[0],buy[1] = -prices[0],max(buy[0],prices[1])
    profit[1] = max(0,prices[1]-prices[0])
    for i in range(2,n):
        buy[i] = max(buy[i-1],profit[i-2]-prices[i])
        profit[i] = max(profit[i-1],buy[i]+prices[i])
    return profit[-1]
```
20ms，10.9MB

### #11
从两头开始向中间扫描

```
def maxArea(height):
    n = len(height)
    if n < 2:
        return 0
    start,end,water = 0,n-1,0
    while start < end:
        h_start,h_end = height[start],height[end]
        if height[start] < height[end]:
            water = max(water,height[start]*(end-start))
            while height[start] <= h_start and start < end:
                start += 1
        else:
            water = max(water,height[end]*(end-start))
            while height[end] <= h_end and start < end:
                end -= 1
    return water
```
### #42
依旧用**相向型双指针算法**，要找到左边最高和右边最高。

```
def trap(height):
    if len(height) < 3:
        return 0
    start,end,water = 0,len(height)-1,0
    s_max,e_max = height[start], height[end]
    while start <= end:
        if s_max < e_max:
            s_max = max(s_max, height[start])
            water += s_max - height[start]
            start += 1
        else:
            e_max = max(e_max, height[end])
            water += e_max - height[end]
            end -= 1
    return water
```
### #334
我自己的解法是20ms，11MB
```
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
```
Discuss里有一写法更为简单，思路相同：

```
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
```
16ms，11.2MB

### #128
比较简单，第一次就20ms，11.2MB

```
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
```
### #164 
用的和上一题一样的思路，第一次就20ms，11.3MB 

```
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
```
这题考察的似乎是 **桶排序(Bucket sort)/基数排序（Radix sort)** 算法,参考了最高票的Discuss，108ms，算法复杂度是O(32n)如下：

```
def radixSort(nums):
    for i in range(31):
        onebucket = []
        zerobucket = []
        needle = 1 << i 
        for j in range(len(nums)):
            if nums[j] & neddle != 0:
                onebucket.append(nums[j])
            else:
                zerobucket.append(nums[j])
        nums = []
        nums += zerobucket
        nums += onebucket 
    return nums 
def maxgap(nums):
    if len(nums) < 2: return 0 
    nums = radixSort(nums)
    res = 0 
    for i in range(1,len(num)):
        res = max(nums[i]-nums[i-1],res)
    return res
```
### #287
这道题先是要证明，这很简单，根据抽屉原理n个抽屉分配n+1个数，必有两个数分配在同一抽屉。接下来要找出这个重复的数我就用了Counter，算法复杂度是o(n)：

```
from collections import Counter
def findDup(nums):
    n = Counter(nums)
    for i in nums:
        if n[i]>1:
            return i
```
实际上，这道题据说花费了Don Knuth 24h才解出来。而且第二问存在四个约束条件。
> 解决本题的主要技巧就是要注意到：重复元素对应于一对下标i != j满足f(i) == f(j).我们的任务就变成寻找一对(i, j).
寻找这个重复值的问题就变成计算机科学界一个广为人知的“环检测”(**cycle detection** or **cycle finding**)问题。给定一个p型序列，在线性时间，只使用常数空间寻找环的起点，这个算法是由Robert Floyd提出的“龟兔”算法(**Floyd's tortoise and the hare algorithm**) 。算法的美妙之处在于只用o(1)的额外存储空间来记录slow指针和fast指针（第一部分），以及finder指针（第二部分）。运行时间复杂度为o(n)
> 1. 第一步，让fast指针前进的速度是slow指针的倍数，当它们第一次相遇时，停下。设环起点距离列表起点为m，环的长度为n，它们相遇时距离环起点的距离为k，则slow走过的距离 i = m+a* n+k, fast走过的距离为2i = m+b* n+k, 距离相减可以得到 i = (b-a)* n, 说明slow和fast走过的距离都是环长的倍数；
> 2. 第二步，让一个指针不动，另一指针移动到列表起点，两者同步移动，则相遇点即为环起点。如slow不动，fast移动到列表起点，更名为finder。当finder到达环起点，即移动m距离时，slow移动了m+i，而i为环长的倍数，也就是说slow也在环起点，所以slow和finder相遇点即为环起点。

```
def findDup1(nums):
    slow = fast = 0
    # keep advancing "slow" by one step and "fast" by two steps
    # until they meet inside the loop 
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break 
    # Start up another pointer from the end of the array and march
    # it forward until it hits the pointer inside the array
    finder = 0
    while True:
        slow = nums[slow]
        finder = nums[finder]
        # If the two hit, the intersection index is the duplicate element
        if slow == finder:
            return finder
```
