# 我记得这有独有的公式，先试试快慢指针吧
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        p1 = head
        p2 = head

        while p2 and p2.next:                      # 这里的边界条件最重要
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                return True
        return False


# 测试(这个好像没法测试)
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def build_with_cycle(vals, pos):
    """把列表建成链表；pos 是环入口下标，-1 表示无环"""
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if pos != -1:
        nodes[-1].next = nodes[pos]      # 尾巴连回 pos，形成环
    return nodes[0]
