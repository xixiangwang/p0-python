class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])                 # 注意这里的self
        right = self.sortArray(nums[mid:])

        return self.merge(left,right)
        
    def merge(self,a,b):
        res = []
        i = 0
        j = 0

        while i<len(a) and j<len(b):
            if a[i] <= b[j]:
                res.append(a[i])
                i += 1
            else:
                res.append(b[j])
                j += 1

        res.extend(a[i:])
        res.extend(b[j:])
        return res

print(Solution().sortArray([5,1,1,2,0,0]))
print(Solution().sortArray([5,2,3,1]))