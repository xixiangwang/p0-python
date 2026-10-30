class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        ans = []
        d = {}
        stack = []

        for i,x in enumerate(nums2):
            while stack and x > stack[-1]:
                t = stack.pop()
                d[t] = x
            stack.append(x)
        while stack:
            t = stack.pop()
            d[t] = -1

        for x in nums1:
            ans.append(d[x])
        
        return ans
            
print(Solution().nextGreaterElement([4,1,2],[1,3,4,2]))
print(Solution().nextGreaterElement([2,4],[1,2,3,4]))
