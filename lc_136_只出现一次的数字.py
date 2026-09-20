# lc_136_只出现一次的数字

# 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
# 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = {}
        for n in nums:
            d[n] = d.get(n,0) + 1
        for k,v in d.items():
            if v == 1:
                return k
print(Solution().singleNumber([4,1,2,1,2]))
print(Solution().singleNumber([2,2,1]))       # 期望 1
print(Solution().singleNumber([1]))           # 期望 1（只有一个元素）



# 这里就是dict两个最重要的操作
# 计数：d[n] = d.get(n,0) + 1
# 遍历：for k,v in d.items():