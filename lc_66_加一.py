# 给定一个表示 大整数 的整数数组 digits，其中 digits[i] 是整数的第 i 位数字。这些数字按从左到右，从最高位到最低位排列。这个大整数不包含任何前导 0。
# 将大整数加 1，并返回结果的数字数组。
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        for i in range(1,len(digits)+1):
            if digits[-i]+1 == 10:
                if i+1 > len(digits):
                    digits[-i] = 0
                    digits.insert(0,1)
                    break
                else:
                    
                    digits[-i] = 0
                    
            else:
                digits[-i] += 1
                break

        return digits

print(Solution().plusOne([1,2,3]))
print(Solution().plusOne([4,3,2,1]))
print(Solution().plusOne([9]))
print(Solution().plusOne([9,9]))