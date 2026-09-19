# # 暴力解
# class Solution(object):
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         for i in range(len(nums)) :
#             for j in range(i+1,len(nums)):
#                 if nums[i] + nums[j] == target:
#                     return [i,j]
# print(Solution().twoSum([2,7,11,15],9))

# dict      很灵活
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}
        for i,n in enumerate(nums):
            need = target-n
            if need in d:
                return [d[need],i]
            d[n] = i
print(Solution().twoSum([2,7,11,15],9))
            
