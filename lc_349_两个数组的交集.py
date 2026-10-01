# =========================== 方法一：set的交集& ======================================
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        nums1_set = set(nums1)
        nums2_set = set(nums2)
        return list(nums1_set & nums2_set)

print(Solution().intersection([1,2,2,1],[2,2]))               # [2]
print(Solution().intersection([4,9,5],[9,4,9,8,4]))           # [9,4],set 是无序的，实际输出可能是 [9, 4] 也可能是 [4, 9]
print(Solution().intersection([1,2,3],[4,5,6]))               # []
