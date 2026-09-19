# ===== lc_1929_数组串联.py =====
# 本文件包含：LC 1929 数组串联（Easy）—— 一行解决：nums + nums（list 相加返回新 list）
# 用到：list 相加 · return
# =============================

class Solution:
    def getConcatenation(self, nums):
        return nums + nums

print(Solution().getConcatenation([1, 2, 1]))   # 期望 [1, 2, 1, 1, 2, 1]