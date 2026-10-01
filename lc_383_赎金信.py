class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        count = {}      # 字典
        for ch in magazine:
            count[ch] = count.get(ch,0) + 1
        
        for ch in ransomNote:
            # if ch not in count:
            #     return False
            # else:
            #     if count[ch] < 1:
            #         return False
            # count[ch] -= 1
        
            # 这里可以更简洁一点，count.get(ch,0)不存在也可以看作0
            if count.get(ch,0) < 1:
                return False
            count[ch] -= 1

        return True

print(Solution().canConstruct('a','b'))             # False
print(Solution().canConstruct('aa','ab'))           # False
print(Solution().canConstruct('aa','aab'))          # True
print(Solution().canConstruct('','abc'))            # True