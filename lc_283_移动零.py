# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
# 请注意 ，必须在不复制数组的情况下原地对数组进行操作。

# 双指针原地改（[0,1,0,3,12]→[1,3,12,0,0]）

# 方法一：暴力解
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero_count = 0
        for n in nums:
            if n == 0:
                zero_count += 1
        
        for i in range(zero_count):
            nums.remove(0)

        for i in range(zero_count):
            nums.append(0)
        
        return nums

nums = [0,1,0,3,12]
Solution().moveZeroes(nums)      # 先调用，让它改 nums
print(nums)                      # 再自己打出来 → [1,3,12,0,0]


# 方法二(简洁版)
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        for n in nums:
            if n == 0:
                nums.remove(0)
                nums.append(0)
# 方法三：快慢指针原地改（[0,1,0,3,12]→[1,3,12,0,0]）
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i],nums[j] = nums[j],nums[i]
                j += 1
nums = [0,1,0,3,12]
Solution().moveZeroes(nums)      # 先调用，让它改 nums
print(nums)                      # 再自己打出来 → [1,3,12,0,0]

