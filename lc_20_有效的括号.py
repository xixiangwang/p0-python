class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        st = []
        is_valid = True
        pairs = {')':'(' , ']':'[' , '}':'{'}        # 这里很重要
        for ch in s:
            if ch in '([{':                          # 简洁
                st.append(ch)
            
            else:
                if not st:
                    return False
                if st[-1] != pairs[ch]:
                    return False
                st.pop()

        return len(st) == 0
print(Solution().isValid("()"))                    
                  
