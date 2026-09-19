# ===== lc_242_有效的字母异位词.py =====
# 本文件包含：LC 242 有效的字母异位词（Easy）—— 计数 dict：s 先加满，t 逐个减，减到负数就 False
# 用到：先比 len · 计数公式 · ch not in count · count[ch] -= 1
# 相关：计数公式 → d5_dict_str_set.py · d5_wordcount.py
# ================================

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        count = {}
        for ch in s:
            count[ch] = count.get(ch,0)+1
        for ch in t:
            if ch not in count:
                return False
            count[ch] -= 1
            if count[ch] < 0:
                return False
        return True
print(Solution().isAnagram("anagram", "nagaram"))