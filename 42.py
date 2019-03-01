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
