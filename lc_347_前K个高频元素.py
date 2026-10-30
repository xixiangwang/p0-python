# 堆，自己写
# ⭐元组的操作很重要，排序都是默认按第一个
import heapq
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 先字典存出现次数,然后组成元组
        d = {}
        for x in nums:
            d[x] = d.get(x,0) + 1
        
        heap = []

        for num,freq in d.items():
            heapq.heappush(heap,(freq,num))
            if len(heap) > k:
                heapq.heappop(heap)
        res = []
        for freq,num in heap:
            res.append(num)
        
        return res

# 测试
s = Solution()
print(s.topKFrequent([1,1,1,2,2,3],2))
print(s.topKFrequent([1],1))
print(s.topKFrequent([1,2,1,2,1,2,3,1,3,2],2))