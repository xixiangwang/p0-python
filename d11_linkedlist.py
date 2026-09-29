# =====================================链表============================================

# 1.定义
class Node:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next              # next 默认 None = 链尾

    def __str__(self):
        return str(self.val)


# 2.手动连起来
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)

head = n1
n1.next = n2
n2.next = n3


# 3.遍历
def dump(head):
    cur = head                       # 用 cur 走，不破坏 head
    while cur:                       # 等价 while cur is not None（真值规则，你学过）
        print(cur.val,end="->")
        cur = cur.next               # 关键的一句：走到下一个节点
    print("None")

dump(n1)



# 4.头插
head = None
for v in [1,2,3]:
    head = Node(v,head)             # 新节点直接指着旧的 head → 它就成了新 head



# 5.尾插
head = None
tail = None
for v in [1,2,3]:
    nd = Node(v)
    if head == None:
        head = nd
        tail = nd
    else:
        tail.next = nd
        tail = nd