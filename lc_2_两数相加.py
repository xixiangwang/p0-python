# 自己写的大部分对，但还是有很多没考虑到，比如两个不等长
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: Optional[ListNode]
        :type l2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(0)
        cur = dummy
        carry = 0       # 进位

        # 感觉这个while最巧妙了，三种情况全包含了
        while l1 or l2 or carry:         ## ③ 三个条件：l1 没走完 / l2 没走完 / 还有进位没落地
            a = l1.val if l1 else 0   # 这里也是非常巧妙
            b = l2.val if l2 else 0

            s = a + b + carry
            carry = s // 10
            cur.next = ListNode(s % 10)

            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        
        return dummy.next