# 堆，示范代码
# 时间复杂度O(n)的话就不能用那些排序了，只能空间换时间，堆
# 思路：维护一个容量为 K 的小顶堆，堆里始终留着"目前最大的 K 个"，
#       堆顶就是这 K 个里最小的那个 —— 正好是"第 K 大"
import heapq                   # ⭐这是小顶堆，标准库里没有大顶堆，但可以都变成负数，就相当于大顶堆了
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for x in nums:
            heapq.heappush(heap,x)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heap[0]


# 测试
s = Solution()
print(s.findKthLargest([3, 2, 1, 5, 6, 4], 2))            # 5
print(s.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))   # 4
print(s.findKthLargest([1], 1))                            # 1
print(s.findKthLargest([2, 1], 2))                         # 1