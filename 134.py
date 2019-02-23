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

def canComplateCircuit1(gas,cost):
    start,overall,accumulate = 0,0,0
    for i in range(gas):
        accumulate += gas[i]-cost[i]
        overall += gas[i]-cost[i]
        if accumulater < 0:
            start,accumulate = i+1,0
    return start if overall>0 else -1

