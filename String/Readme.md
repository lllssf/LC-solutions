# String
## Primary
### #344
用同样的算法，python太慢，但因为题目限制，不能直接用
```
return s[::-1]
```
### #392
用两个指针or用iterator & generator
第二个不太好理解，拆解一下大概可以写为：
```
def is_subsequence(a,b):
  b = iter(b)
  for i in a:
    while True:
      try:
        val = next(b)
        if val == i:
          yield True
          break
      except StopIteration:
        yield False
        return
all(is_subsequence('123','12345'))
```
