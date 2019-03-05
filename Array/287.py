from collections import Counter
def findDup(nums):
    n = Counter(nums)
    for i in nums:
        if n[i]>1:
            return i

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
            return slow
