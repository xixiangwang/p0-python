# 例题，滑动窗口，确实没思路
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        left = 0
        total = 0
        best = n+1

        for right in range(n):
            total += nums[right]

            while total >= target:
                best = min(best,right-left+1)
                total -= nums[left]
                left += 1
        if best == n+1:
            return 0
        else:
            return best

print(Solution().minSubArrayLen(7,[2,3,1,2,4,3]))
print(Solution().minSubArrayLen(11,[1,1,1,1,1,1,1,1]))



# ======= 为什么是 O(n)​? ====================
# 看着是 for 里套 while，很像双重循环。但数一下代价：right 每个元素只加进窗口一次，left 每个元素只退出窗口一次，
# 两个指针加起来最多各走 n 步，总操作数约 2n。这是均摊分析，不是表面看的 O(n²)。