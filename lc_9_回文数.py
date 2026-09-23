# 一行版：
class Solution():
    def isPalindrome(self,x):
        s = str(x)
        return s == s[::-1]

print(Solution().isPalindrome(121))
print(Solution().isPalindrome(233))
print(Solution().isPalindrome(1))
print(Solution().isPalindrome(-121))
print(Solution().isPalindrome(10))    # 期望 False  ← "10" vs "01"
print(Solution().isPalindrome(0))     # 期望 True