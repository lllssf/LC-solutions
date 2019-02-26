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

def maxProfit(prices):
    return sum(x[0]-x[1] for x in zip(prices[1:],prices[:-1]) if x[0]>x[1] )
