# 看答案写的，思路其实不难，主要是想要遍历，那只能是用right，然后再变换left。这里有个新知识点ord(),ascii编码。
# 还有两个need和win可以直接比
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        n1 = len(s1)
        if n1 > len(s2):
            return False
        
        need = [0] * 26
        win = [0] * 26

        for ch in s1:
            need[ord(ch) - ord('a')] += 1

        left = 0
        for right in range(len(s2)):
            win[ord(s2[right]) - ord('a')] += 1

            if right - left + 1 > n1:
                win[ord(s2[left]) - ord('a')] -= 1
                left += 1
            
            if right - left + 1 == n1 and need == win:
                return True
        return False


print(Solution().checkInclusion("ab","eidbaooo"))
print(Solution().checkInclusion("ab","eidboaoo"))