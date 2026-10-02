# 刚上来思路是递归，但下标难以实现。
# 后来想到左右指针，思路大体对，但忽略了几个细节
# 1.当target < nums[mid]时，应该left=mid-1，我写成了left=mid。另一种情况也是
# 2.if-else语句语法错误。我写成了if-if-else,正确写法是if-elif-else
# 应该是有两个指针指向区间
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        
        while left <= right:
            mid = (left + right) // 2              # 向下取整
            if target < nums[mid]:
                right = mid-1
            elif target > nums[mid]:
                left = mid+1
            else:
                return mid
        return -1



# ---------------- 本地自测 ----------------
print(Solution().search([-1,0,3,5,9,12], 9))          # 4
print(Solution().search([-1,0,3,5,9,12], 2))          # -1
print(Solution().search([], 5))                   # -1
