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
