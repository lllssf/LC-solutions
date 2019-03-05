def maxprofit(prices):
    n = len(prices)
    if n < 2:
        return 0
    buy,profit = [float('-inf')]*n,[0]*n
    buy[0],buy[1] = -prices[0],max(buy[0],-prices[1])
    profit[1] = max(0,prices[1]-prices[0])
    for i in range(2,n):
        buy[i] = max(buy[i-1],profit[i-2]-prices[i])
        profit[i] = max(profit[i-1],buy[i]+prices[i])
    return profit[-1]
