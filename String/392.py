# 双指针
def is_subsequence(s,t):
  p1 = 0
  for p2 in range(len(t)):
    if s[p1] == t[p2]:
      p1 += 1
      if p1 == len(s): return True
  return False
  
# 利用生成器和迭代器
def is_subsequence(s,t):
  t = iter(t)
  return all((i in t) for i in s)
