# =================================解法一：新建节点，头插法
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        # 头插法
        jiahead = None
        while head:
            jiahead = ListNode(head.val,jiahead)
            head = head.next
        return jiahead



# ===============================标准解法（就地反转）================================
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

    def reverseList_insert(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        prev = None
        cur = head
        while cur:
            nxt = cur.next       # 先存住下一个
            cur.next = prev
            prev = cur
            cur = nxt
        return prev


# ---------------- 本地自测 ----------------
def build(xs):
    head = None
    for v in reversed(xs):          # reversed() 内置函数：反过来遍历
        head = ListNode(v, head)
    return head


def dump(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


for xs in [[1,2,3,4,5], [1,2], []]:
    print(xs, "->", dump(Solution().reverseList(build(xs))),
          "| 头插 ->", dump(Solution().reverseList_insert(build(xs))))

        

