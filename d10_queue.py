# ==================================队列（Queue）—— 先进先出 FIFO==============================
# 从队尾进，从队头出。先来的先走。
# list 删队头要挪全部元素（O(n)），deque.popleft() 是 O(1)

# 1.用 list 也能做，但有个性能雷：
# pop(0) 能拿到最早的，但很慢
q = []
q.append(1)             # 从队尾进
q.append(2)
q.append(3)
print(q.pop(0))         # 1   ← 必须 pop(0)：弹【第一个】才是队头
print(q.pop())          # 3  这样好像是队尾


# ⭐2.deque 怎么用（新语法）
# 当队列用：append 进 + popleft 出。
# 当栈用：append 进 + pop 出。

from collections import deque  

d = deque([1,2,3])

d.append(4)               # 右进 → deque([1, 2, 3, 4])
print(d)                  
d.appendleft(0)           # 左进 → deque([0, 1, 2, 3, 4])
print(d)
print(d.pop())            # 右出 → 4
print(d.popleft())        # 左出 → 0




