def getHint(secret,guess):
    bull = 0
    cow = 0
    index = list(range(len(secret)))
    # calculate the value of bulls and get rid of its index
    for i in index[:]:
        if secret[i] == guess[i]:
            bull += 1
            index.remove(i)
    # get the value of cows
    for i in index[:]:
        for j in range(len(guess)):
            if secret[i] == guess[j] and j in index: # to avoid being counted before
                cow += 1
                index.remove(j)
                break
    return '%sA%sB' %(bull,cow)

print(getHint('1123','0111'))

from collections import Counter
def getHint1(secret,guess):
    '''
    use Counter to count guess and secret and sum their overlap.
    use zip to counter bulls
    '''
    s,g=Counter(secret),Counter(guess) #return a dict
    bull = sum(i == j for i,j in zip(secret,guess))
    cow = sum((s & g).values()) - bull
    return '%sA%sB' %(bull,cow)

print(getHint1('1123','0111'))

def getHint2(secret,guess):
    bull = 0
    cow = 0
    counts = {} #caiculate the counts of s
    for i,s in enumerate(secret):
        if s == guess[i]:
            bull += 1
        else:
            counts[s] = counts.get(s,0) + 1
    for i,s in enumerate(secret):
        if guess[i]!= s and counts.get(guess[i],0)!=0:
            cow += 1
            counts[guess[i]] -= 1
    return '%sA%sB' %(bull,cow)

print(getHint2('1123','0111'))
