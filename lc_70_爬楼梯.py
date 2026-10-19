# dp序列型问题
# lc_70 爬楼梯
# 题目：一次可以爬 1 级或 2 级，求爬到第 n 级有多少种方法
# 钥匙：站在第 n 级往回看，最后一步只能来自 n-1（迈 1 级）或 n-2（迈 2 级）

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        
        dp = [0] * (n+1)       # 因为从1开始
        dp[1] = 1
        dp[2] = 2

        for i in range(3,n+1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]

# 自测
for n in [1, 2, 3, 4, 5]:
    print(n, Solution().climbStairs(n))