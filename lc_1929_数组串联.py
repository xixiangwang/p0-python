# ===== lc_1929_数组串联.py =====
# 本文件包含：LC 1929 数组串联（一行：nums + nums）
# =============================
class Solution:
    def getConcatenation(self, nums):
        return nums + nums

print(Solution().getConcatenation([1, 2, 1]))   # 期望 [1, 2, 1, 1, 2, 1]