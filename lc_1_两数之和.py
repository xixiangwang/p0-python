# ===== lc_1_两数之和.py =====
# 本文件包含：LC 1 两数之和（Easy）—— ① 暴力双重循环版（已注释保留）② 一趟 dict「先查再存」版
# 用到：enumerate 拿下标 · dict 存「见过的数 → 下标」· need in d
# 相关：面试出现率最高的一道 Easy；dict 计数同族 → lc_242_有效的字母异位词.py
# ==========================

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
            
