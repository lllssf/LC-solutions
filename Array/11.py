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
