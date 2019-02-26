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
