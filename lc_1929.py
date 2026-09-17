# lc_1929.py
class Solution:
    def getConcatenation(self, nums):
        return nums + nums

print(Solution().getConcatenation([1, 2, 1]))   # 期望 [1, 2, 1, 1, 2, 1]