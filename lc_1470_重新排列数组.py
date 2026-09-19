# ===== lc_1470_重新排列数组.py =====
# 本文件包含：LC 1470 重新排列数组（Easy）—— 前半后半交替：res.append(nums[i]) + res.append(nums[n+i])
# 用到：range(n) · append · 索引 · 新建空 list 累积
# ===============================

# LC 1470 重新排列数组
class Solution(object):
    def shuffle(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: List[int]
        """
        res = []
        for i in range(n):
            res.append(nums[i])
            res.append(nums[n+i])
        return res
print(Solution().shuffle([2,5,1,3,4,7],3))    