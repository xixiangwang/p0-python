# ============================ 简易哈希表（链地址法）============================

# hash("张三") 能算，是因为字符串在计算机里就是"一串字符编号"，编号能拍成一个整数。​
# % size 再把这个大整数（可能为负）拉进 0 ~ size-1 的桶号范围。​
# 但哈希值每次运行都会变（Python 故意随机化）→ 桶编号不可预测 → 验收只验"存取正确"，别验桶号
class Node:
    """链表的节点：存一个键值对"""
    def __init__(self,key,value):
        self.key = key
        self.value = value
        self.next = None


class MyHashTable:
    def __init__(self,size=7):
        self.size = size
        # 每个格子挂一条链，初始都是空的（None）
        self.buckets = [None] * size

    def _index(self,key):
        """哈希函数：算出这个 key 该放哪个格子"""
        return hash(key) % self.size

    def put(self,key,value):                   # 有点难理解
        """存入 / 更新"""
        i = self._index(key)
        cur = self.buckets[i]

        while cur:                             # 先看链上有没有这个 key
            if cur.key == key:
                cur.value = value              # 有 → 更新
                # 它在这里的作用不是"返回结果"，是 “提前跑掉，不许再往下执行” ​。
                return
            
            cur = cur.next
        # 没有 → 头插一个新节点（就是昨天学的头插法）
        nd = Node(key,value)
        nd.next = self.buckets[i]
        self.buckets[i] = nd


    def get(self,key):
        """取值，找不到返回 None"""
        i = self._index(key)
        cur = self.buckets[i]

        while cur:
            if cur.key == key:
                return cur.value
            cur = cur.next

        return None


    def remove(self,key):
        """删除"""
        i = self._index(key)
        cur = self.buckets[i]
        prev = None

        while cur:
            if cur.key == key:
                if prev is None:
                    self.buckets[i] = cur.next      # 删的是链头
                else:
                    prev.next = cur.next            # 跳过它
                return True

            prev = cur
            cur = cur.next

        return False



    def keys(self):
        """返回所有 key"""
        out = []
        for cur in self.buckets:
            while cur:
                out.append(cur.key)
                cur = cur.next

        return out




    # ---------------- 验收 ----------------
ht = MyHashTable()
ht.put("name", "张三")
ht.put("score", 85)
ht.put("city", "武汉")

print(ht.get("name"))       # 张三
print(ht.get("score"))      # 85
print(ht.get("nothing"))    # None

ht.put("score", 90)         # 更新
print(ht.get("score"))      # 90

print(ht.remove("city"))    # True
print(ht.get("city"))       # None
print(ht.remove("city"))    # False  （已经删了）

print(sorted(ht.keys()))    # ['name', 'score']   ← 用 sorted 让顺序稳定


# ---------------- 冲突测试：强行让多个 key 进同一个桶 ----------------
print()
t = MyHashTable(size=1)          # size=1 → 所有 key 都算到 0 号桶
for k in ["a", "b", "c", "d"]:
    t.put(k, k.upper())

print("链上:", t.keys())          # ['d', 'c', 'b', 'a']（头插，顺序是反的）
print("删链中间的 b ->", end=" ")
print(t.remove("b"))
print("删完:", t.keys())


