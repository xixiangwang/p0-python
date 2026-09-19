# ===== lc_1480_一维数组的动态和.py =====
# 本文件包含：LC 1480 一维数组的动态和（Easy）—— 前缀和：边遍历边累加 running，再 append 进结果
# 用到：for 遍历 · 累加 · append · 返回新 list
# =================================

class Solution:
    def runningSum(self, nums):
        running = 0
        res = []
        for x in nums:
            running += x
            res.append(running)
        return res

print(Solution().runningSum([1, 2, 3, 4]))      # 期望 [1, 3, 6, 10]