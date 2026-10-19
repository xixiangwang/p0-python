# dp序列型问题（同lc_70爬楼梯）
# 先把每个位置能得到的最大值dp找出来，然后再输出max(dp)
# dp:以这个位置为最后，能取到的最大值
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1,n):
            if dp[i-1] + nums[i] < nums[i]:
                dp[i] = nums[i]
            else:
                dp[i] = dp[i-1] + nums[i]
        return max(dp)

# 测试
print(Solution().maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
print(Solution().maxSubArray([1]))
print(Solution().maxSubArray([5,4,-1,7,8]))