# 和LC300很像，但这题是连续，就简单在这里
class Solution(object):
    def findLengthOfLCIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        dp = [0] * n

        dp[0] = 1

        for i in range(1,n):
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1] + 1
            else:
                dp[i] = 1
        
        return max(dp)

# 测试
print(Solution().findLengthOfLCIS([1,3,5,4,7]))
print(Solution().findLengthOfLCIS([2,2,2,2,2]))