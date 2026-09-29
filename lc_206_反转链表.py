# =================================解法一：新建节点，头插法
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
class Solution(object):
    def reverseList(self, head):
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

        

