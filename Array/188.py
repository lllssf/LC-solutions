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
