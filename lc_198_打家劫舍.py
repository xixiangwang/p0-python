# dp序列型问题(同53，70)
# dp[i]:到i号房为止，最多是多少(感觉和爬楼梯很像)
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)

        if n == 1:
            return nums[0]
        dp = [0] * n

        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])

        for i in range(2,n):
            dp[i] = max(dp[i-2]+nums[i],dp[i-1])
        return dp[n-1]

# 测试
print(Solution().rob([0]))
print(Solution().rob([1,2,3,1]))
print(Solution().rob([2,7,9,3,1]))
