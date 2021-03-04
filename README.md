# LeetCode刷题记录

[toc]

按照网上给出的分类和刷题建议整理。

## Array

### Primary

#### #27 [Remove Element](https://leetcode.com/problems/remove-element/)

```python
# -*- coding: utf-8 -*-
def removeElement(nums,val):
    #ERROR:修改遍历的列表
    for x in nums[:]:
        if x == val:
            nums.remove(x)
    return len(nums)
```

容易出现的问题就是：

```python
for x in nums:
```

> 我们通常尽量避免修改一个正在进行遍历的列表，可以使用**切片操作克隆这个list**来避免这个问题（浅拷贝）

#### #26 [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/)

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        while i+1 < len(nums):
            if nums[i] == nums[i+1]:
                nums.pop(i)
            else:
                i += 1
        return len(nums)
```

#### #80 [Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/)

因为都是有序数列所以和上一题思路一样

```python
def removeDuplicates(nums):
    dupnums = [x for x in nums]
    i = 0
    for x in dupnums[:]:
        if x in dupnums[i+2:]:
            nums.remove(x)
        i = i+1
        if i == len(dupnums)-2:
            return nums
```

#### #189 [Rotate Array](https://leetcode.com/problems/rotate-array/description/)

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        if k <= 0 or len(nums) <= 1:
            return 
        nums.reverse()
        nums[:k] = nums[k-1::-1]
        nums[k:] = nums[:k-1:-1]
```

要注意的是，reverse()不能反转部分数组，要想反转部分可以使用切片。

#### #41 [First Missing Positive](https://leetcode.com/problems/first-missing-positive/description/)

```python
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
```

这个在LeetCode测试是16ms，快于100%

#### #299 [Bulls and Cows](https://leetcode.com/problems/bulls-and-cows/)

这道题开始忘了用dict，在计算cow的时候很笨的用了两层循环

```python
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

```python
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

```python
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

#### #134 [ Gas Station](https://leetcode.com/problems/gas-station/description/)

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

#### #274 [ H-Index](https://leetcode.com/problems/h-index/description/)

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

#### #275 [ H-Index II](https://leetcode.com/problems/h-index-ii/description/)

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


#### #217 [ Contains Duplicate](https://leetcode.com/problems/contains-duplicate/description/)

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

#### #55 [ Jump Game](https://leetcode.com/problems/jump-game/description/)

> **广度优先搜索BFS**(breadth-first search) 适用于解决“**最短路径问题**”(shortest-path problem)。解决这类问题：
>
> 1. 使用图来建立问题模型
> 2. 使用广度优先搜索解决问题  

>广度优先搜索算法可帮助回答两类问题：
>
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

#### #45 [ Jump Game II](https://leetcode.com/problems/jump-game-ii/description/)

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

#### #121 [ Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

```
def maxprofit(prices):
    buy,profit = float('inf'),0
    for p in prices:
        buy = min(buy,p)
        profit = max(profit, p-buy)
    return profit
```

#### #122 [ Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/)

维护buy和sale的日期，如果某日的价格小于当前sale日的价格，则出售，且buy和sale的价格更新为当日，若某日价格大于当前sale日的价格，则sale日期更新为当日。循环结束记得还要抛售。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy, sale, profit = 0, 0, 0
        for i in range(1, len(prices)):
            if prices[i] < prices[sale]:
                profit += prices[sale] - prices[buy]
                buy, sale = i, i
            if prices[i] > prices[buy]:
                sale = i
        profit += prices[sale] - prices[buy]
        return profit
```

还有比较简单的写法：

```python
def maxProfit(prices):
    return sum(x[0]-x[1] for x in zip(prices[1:],prices[:-1]) if x[0]>x[1] )
```

#### #123 [ Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/)

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

#### #188 [ Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/)

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

#### #309 [ Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/)

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

#### #11 [ Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/)

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

#### #42 [ Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/description/)

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

#### #334 [ Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/description/)

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

#### #128 [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)

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

#### #164  [ Maximum Gap](https://leetcode.com/problems/maximum-gap/description/)

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

#### #287 [ Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/)

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
> 寻找这个重复值的问题就变成计算机科学界一个广为人知的“环检测”(**cycle detection** or **cycle finding**)问题。给定一个p型序列，在线性时间，只使用常数空间寻找环的起点，这个算法是由Robert Floyd提出的“龟兔”算法(**Floyd's tortoise and the hare algorithm**) 。算法的美妙之处在于只用o(1)的额外存储空间来记录slow指针和fast指针（第一部分），以及finder指针（第二部分）。运行时间复杂度为o(n)
>
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

### Medium

#### #4 [ Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/description/)

参考了Discuss关于中位数的思考：

```
def findMedianSortedArrays(nums1,nums2):
    m,n = len(nums1),len(nums2)
    # Ensure j is non-negative
    if m > n:
        m,n,nums1,nums2 = n,m,nums2,nums1

    imin, imax = 0,m
    while imin<=imax:
        # Ensure len(left)==len(right)
        i = (imin+imax)/2
        j = (m+n+1)/2 - i
        # If i is too small
        if i < m  and nums1[i] < nums2[j-1]:
            imin += 1
        # If i is too big
        elif i > 0 and nums1[i-1] > nums2[j]:
            imax -= 1
        else:
            # i is perfect
            if i==0 : max_of_left = nums2[j-1]
            elif j == 0: max_of_left = nums1[i-1]
            else: max_of_left = max(nums1[i-1],nums2[j-1])
            # (m+n) is odd
            if (m+n) % 2 == 1: return max_of_left
            # (m+n) is even
            if i == m: min_of_right = nums2[j]
            elif j == n: min_of_right = nums1[i]
            else: min_of_right = min(nums1[i],nums2[j])

            return (max_of_left + min_of_right) / 2.0
```

#### #321 [ Create Maximum Number](https://leetcode.com/problems/create-maximum-number/description/)

#### #327 [ Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/description/)

#### #289 [ Game of Life](https://leetcode.com/problems/game-of-life/description/)

### Counter

#### #239 [ Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/)

#### #350 [Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/description/)

取两个数组的交集，其一可以使用计数的方式，利用collections.Counter的哈希对象计数：

```python
from collections import Counter
def solutions(nums1, nums2):
    if not (nums1 and nums2):
        return []
    counts1 = Counter(nums1)
    res = []
    for i in nums2:
        if counts1[i] > 0:
            res.append(i)
            counts1[i] -= 1
    return res
```

其二可以使用双指针对于排序后的数组取交集，即对应元素较小的指针向后移一位：

```python
def solution(nums1, nums2):
    if not (nums1 and nums2):
        return []
    p1, p2, res = 0, 0, []
    while p1<len(nums1) and p2<len(nums2):
        if nums1[p1] == nums2[p2]:
            res.append(nums1[p1])
            p1 += 1
            p2 += 1
        elif nums1[p1] < nums2[p2]:
            p1 += 1
        else:
            p2 += 1
    return res
```

## String

### Primary

#### #28 [Implement strStr()](https://leetcode.com/problems/implement-strstr/description/)

#### #14 [ Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/description/)

将第一个字符串初始化为公共前缀，利用re.match与其他字符串进行比较，若匹配失败则将公共前缀的最后一个字符去掉，直到公共前缀为空或者全部match。注意，当只有一个元素时，公共前缀为该元素而不是"":

```python
import re
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        res = strs[0]
        for i in strs[1:]:
            while re.match(res, i) == None:
                res = res[:-1]
                if not res:
                    return ""     
        return res
```

或者用单指针在第一个字符串上移动，若当前元素与每个字符串对应位置元素相同，则公共前缀纳入该元素，出现不同则直接停止返回当前最大公共前缀：

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        res = min(strs,key=len)
        for i,s in enumerate(res):
            for string in strs:
                if s != string[i]:
                    return res[:i]
        return res
```

还有一个避免了双循环的很机智的解法：

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        s1 = min(strs)
        s2 = max(strs)
        for i,s in enumerate(s1):
            if s != s2[i]:
                return s1[:i]
        return s1
```





#### #58 [ Length of Last Word](https://leetcode.com/problems/length-of-last-word/description/)

#### #387 [First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/description/)

#### #383 [ Ransom Note](https://leetcode.com/problems/ransom-note/description/)

#### #344 [ Reverse String](https://leetcode.com/problems/reverse-string/description/)

#### #151 [ Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/description/)

#### #186 [Reverse Words in a String II](https://leetcode.com/problems/reverse-words-in-a-string-ii/description/)

#### #345 [Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/description/)

#### #205 [ Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/description/)

#### #293 [Flip Game](https://leetcode.com/problems/flip-game/description/)

#### #294 [ Flip Game II](https://leetcode.com/problems/flip-game-ii/description/)

#### #290 [Word Pattern](https://leetcode.com/problems/word-pattern/description/)

#### #242 [ Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)

#### #49 [ Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)

#### #249 [Group Shifted Strings](https://leetcode.com/problems/group-shifted-strings/description/)

#### #87 [ Scramble String](https://leetcode.com/problems/scramble-string/description/)

#### #179 [Largest Number](https://leetcode.com/problems/largest-number/description/)

#### #6 [ ZigZag Conversion](https://leetcode.com/problems/zigzag-conversion/description/)

主要是找规律，然后利用字典按规律存储结果。

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) <= numRows or numRows <= 1:
            return s
        res = {}
        for i in range(len(s)):
            m = i % (2*numRows-2)
            if m <= numRows-1:
                if m in res:
                    res[m] += s[i]
                else:
                    res[m] = s[i]
            else:
                m = (2*numRows-2) - m
                res[m] += s[i]
        s = ""
        for i in res:
            s += res[i]
        return s
```

同样的思想，更简洁的写法：

```python
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s

        L = [''] * numRows
        index, step = 0, 1

        for x in s:
            L[index] += x
            if index == 0:
                step = 1
            elif index == numRows -1:
                step = -1
            index += step

        return ''.join(L)
```



#### #161 [ One Edit Distance](https://leetcode.com/problems/one-edit-distance/)

#### #38 [ Count and Say](https://leetcode.com/problems/count-and-say/description/)

#### #358 [ Rearrange String k Distance Apart](https://leetcode.com/problems/rearrange-string-k-distance-apart/description/)

#### #316 [Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/description/)

#### #271 [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/description/)

#### #168 [ Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/description/)

#### #171 [ Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/description/)

#### #13 [ Roman to Integer](https://leetcode.com/problems/roman-to-integer/description/)

#### #12 [ Integer to Roman](https://leetcode.com/problems/integer-to-roman/description/)

#### #273 [ Integer to English Words](https://leetcode.com/problems/integer-to-english-words/description/)

#### #246 [Strobogrammatic Number](https://leetcode.com/problems/strobogrammatic-number/description/)

#### #247 [ Strobogrammatic Number II](https://leetcode.com/problems/strobogrammatic-number-ii/description/)

#### #248 [Strobogrammatic Number III](https://leetcode.com/problems/strobogrammatic-number-iii/description/)

### Medium

#### #157 [Read N Characters Given Read4](https://leetcode.com/problems/read-n-characters-given-read4/description/)

#### #158 [Read N Characters Given Read4 II - Call multiple times](https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/description/)

#### #68 [ Text Justification](https://leetcode.com/problems/text-justification/description/)

#### #65 [ Valid Number](https://leetcode.com/problems/valid-number/description/)

### Parentheses

#### #20 [ Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)

用堆栈来解决



## Math

### Primary

#### #7 [Reverse Integer](https://leetcode.com/problems/reverse-integer/description/)

#### #165 [Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/description/)

#### #66 [ Plus One](https://leetcode.com/problems/plus-one/description/)

转换成整数加一

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if not digits:
            return digits
        num = ''.join([str(i) for i in digits])
        num = int(num) + 1
        digits = [int(i) for i in str(num)]
        return digits
```



#### #8 [String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/description/)

#### #258 [ Add Digits](https://leetcode.com/problems/add-digits/description/)

#### #67 [Add Binary](https://leetcode.com/problems/add-binary/description/)

#### #43 [ Multiply Strings](https://leetcode.com/problems/multiply-strings/description/)

#### #29 [ Divide Two Integers](https://leetcode.com/problems/divide-two-integers/description/)

#### #69 [Sqrt(x)](https://leetcode.com/problems/sqrtx/description/)

#### #50 [Pow(x, n)](https://leetcode.com/problems/powx-n/description/)

#### #367 [Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/description/)

#### #365 [Water and Jug Problem](https://leetcode.com/problems/water-and-jug-problem/description/)

#### #204 [Count Primes](https://leetcode.com/problems/count-primes/description/)

### SUM

#### #1 [ Two Sum](https://leetcode.com/problems/two-sum/description/)

将数组利用字典存储，快速访问下标：

```python
def twoSum(nums, target):
    d = {}
    for i,n in enumerate(nums):
        if target-n in d:
            return sorted([i, d[target-n]])
        d[n] = i
```



#### #167 [ Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)

#### #15 [ 3Sum](https://leetcode.com/problems/3sum/description/)

重点在于要先对数组进行排序，然后利用指针的思想解决问题。

```python
class Solution:
    def threeSum(self, nums):
        if len(nums) <= 2:
            return []
        nums = sorted(nums)
        res = []
        for i in range(len(nums)-2):
            if i>0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] > 0:
                return res
            left = i + 1
            right = len(nums) - 1
            while left < right:
                n = nums[i] + nums[left] + nums[right]
                if n == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif n > 0:
                    right -= 1
                else:
                    left += 1
        return res
```

#### #16 [ 3Sum Closest](https://leetcode.com/problems/3sum-closest/description/)

#### #259 [ 3Sum Smaller](https://leetcode.com/problems/3sum-smaller/description/)

#### #18 [ 4Sum](https://leetcode.com/problems/4sum/description/)

## Tree 

### Primary

#### #144 [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

#### #94 [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)

#### #145 [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)

#### #102 [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

### Preorder

#### #100 [ Same Tree](https://leetcode.com/problems/same-tree/description/)

#### #101 [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/)

#### #226 [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/)

#### #257 [ Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/description/)

#### #112 [ Path Sum](https://leetcode.com/problems/path-sum/description/)

#### #113 [Path Sum II](https://leetcode.com/problems/path-sum-ii/description/)

#### #129 [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/description/)

#### #298 [ Binary Tree Longest Consecutive Sequence](https://leetcode.com/problems/binary-tree-longest-consecutive-sequence/description/)

#### #111 [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/)

### Postorder

#### #104 [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/)

#### #110 [ Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/description/)

#### #124 [ Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/)

#### #250 [ Count Univalue Subtrees](https://leetcode.com/problems/count-univalue-subtrees/description/)

#### #366 [ Find Leaves of Binary Tree](https://leetcode.com/problems/find-leaves-of-binary-tree/description/)

#### #337 [ House Robber III](https://leetcode.com/problems/house-robber-iii/description/)

## Linked List

### Primary

#### #206 [ Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)

反转链表是最常考的题目，思维简单直接，主要考代码实现。

```python
class Solution:
    def reverseList(self, head):
        cur, prev = head, None
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        return prev
```

#### #141 [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)

#### #24 [ Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/)

## Dynamic Programming

### 1-Dimention

#### #70 [ Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/)

动态规划问题主要是找对状态的定义以及DP状态转移方程，从结果开始递推：

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        res = {}
        res[1] = 1
        res[2] = 2
        if n > 2:
            for i in range(3, n+1):
                res[i] = res[i-1] + res[i-2]
        return res[n]
```

利用python语言特点，可简化为：

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        prev, cur = 1, 1
        for _ in range(1, n):
            prev, cur = cur, cur+prev
        return cur
```

#### #53 [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/)

这道题很容易找到状态转移方程：$opt(n) = max(opt(n-1) + nums[n], nums[n]) $,但关键在于最终返回的不是opt(n)，而是$max(opt(1), opt(2), ..., opt(n))$

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        res = [nums[0]]
        for i in range(1,len(nums)):
            if res[i-1] > 0:
                res.append(res[i-1] + nums[i])
            else:
                res.append(nums[i])
        return max(res)
```

进一步，可以简化为（用max()比条件判断更占用内存）：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        cur_sum = max_sum = nums[0]
        for i in nums[1:]:
            cur_sum = max(i, cur_sum + i)
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

#### #120 [Triangle](https://leetcode.com/problems/triangle/description/)

1. 递归（回溯），时间复杂度是$O(2^N)$
2. 贪心会陷入局部最优，可能会错过全局最优
3. DP：
   1. 定义状态：dp[i,j]表示从最底层走到(i,j)位置路径和的最小值，则本题的结果存在dp[0,0]中
   2. DP方程：$dp[i,j] = min(dp[i+1, j], dp[i+1, j+1]) + triangle(i,j)$,
   3. 初始状态：$dp[m-1,j] = triangle(m-1,j)$
   4. 结果：$dp[0][0]$

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 1:
            return triangle[0][0]
        m = len(triangle)
        dp = [[triangle[i][j] for j in range(len(triangle[i]))] for i in range(m)]
        for i in range(m-2,-1,-1):
            for j in range(len(triangle[i])):
                dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + triangle[i][j]
        return dp[0][0]
```

用二维数组浪费空间，实际每次只需要存储一层的结果，用一位数组即可：

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 1:
            return triangle[0][0]
        dp = triangle[-1]
        for i in range(len(triangle)-2,-1,-1):
            for j in range(len(triangle[i])):
                dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
        return dp[0]
```



#### #152 [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)

DP不熟悉的时候很容易将该题的状态定义为1d数组，然后状态转移方程定义为$dp[i] = dp[i-1] * nums[i]$时就会发现对于$nums[i]$为正/负数的情况无法区分，因为$nums[i]$为正时应与max_product相乘，而$nums[i]$为负时应与最小值相乘。因此状态应具备存储max_product & min_product的能力。

1. 定义状态：$dp[i][2]$；$dp[i][0]$表示第i位时的最大乘积，$dp[i][1]$表示i位的最小乘积

2. DP方程：$dp[i][0] = dp[i-1][0] * nums[i], if(nums[i]>=0)$

   ​			   $dp[i][0] = dp[i-1][1] * nums[i], if (nums[i]<0)$

   ​			   $dp[i][1] = dp[i-1][1] * nums[i], if (nums[i]>=0)$

   ​			   $dp[i][1] = dp[i-1][0] * nums[i], if (nums[i]<0)$

3. 初始状态：$dp[0][0] =nums[0], dp[1][1] = nums[0]$

4. 结果：$max(dp[i][0])$

为了减少内存使用，不需要用一个数组去存储结果，只需要保存当前最大乘积、最小乘积和全局最大乘积即可：

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = min_prod = max_prod = nums[0]
        for i in nums[1:]:
            if i >= 0:
                max_prod = max(max_prod * i, i)
                min_prod = min_prod * i
            else:
                max_prod, min_prod = min_prod * i, min(max_prod * i, i)
            res = max(max_prod, res)
        return res
```

#### #300 [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/description/)

1. 定义状态：dp[i]

2. DP方程：$dp[i] = max(dp[j]) + 1, 0<i<n, 0<=j<i$

3. 初始状态：$dp[0] = 1$

4. 结果：$max(dp[i])$

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]
        for i in range(1, len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)
```

算法时间度是$O(N)$，

