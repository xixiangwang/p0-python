# lc_344_反转字符串

# 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。
# 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        for i in range(len(s)//2):
            s[i],s[-i-1] = s[-i-1],s[i]

s1 =  ["h","e","l","l","o"]       
Solution().reverseString(s1)
print(s1)

s2 = ["h","l","o","w"]
Solution().reverseString(s2)
print(s2)

# s[::-1]本质是创建了一个数组，空间不是O(1)