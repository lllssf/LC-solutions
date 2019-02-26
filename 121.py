def maxprofit(prices):
    buy,profit = float('inf'),0
    for p in prices:
        buy = min(buy,p)
        profit = max(profit, p-buy)
    return profit
