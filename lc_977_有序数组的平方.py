class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sorted_nums = sorted(nums,key=abs)
        return [n*n for n in sorted_nums]

print(Solution().sortedSquares([-4,-1,0,3,10]))



# ============================双指针=============================
class Solution2(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        res = [0] * n
        i = 0
        j = n-1
        k = n-1
        while i <= j:
            if abs(nums[i]) <= abs(nums[j]):
                res[k] = nums[j] * nums[j]
                j -= 1
            else:
                res[k] = nums[i] * nums[i]
                i += 1
            
            k -= 1
        
        return res

print(Solution2().sortedSquares([-4,-1,0,3,9]))