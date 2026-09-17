# lc_1480.py
class Solution:
    def runningSum(self, nums):
        running = 0
        res = []
        for x in nums:
            running += x
            res.append(running)
        return res

print(Solution().runningSum([1, 2, 3, 4]))      # 期望 [1, 3, 6, 10]