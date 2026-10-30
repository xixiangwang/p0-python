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


    # 不用dp，本质是计数
    def findLengthOfLCIS2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = 1
        best = 1
        for i in range(len(nums)-1):
            if nums[i+1] > nums[i]:
                count += 1
                best = max(best,count)
            else:
                count = 1
        return best

# 测试
print(Solution().findLengthOfLCIS([1,3,5,4,7]))
print(Solution().findLengthOfLCIS([2,2,2,2,2]))
print(Solution().findLengthOfLCIS2([1,3,5,4,7]))
print(Solution().findLengthOfLCIS2([2,2,2,2,2]))


# 这题不需要dp也可以，实质是计数，空间只要O(1)


# sk-ws-H.PXMDMRH.Vxwm.MEUCIQDnHN_C0lVK0HfsaVR12fWwFU0kC9V-dFgERDTk0ndRwgIgIF-PxrJw4W7Pgfas0-jZ_Dly206FWu9I-5-sPJxaxxo