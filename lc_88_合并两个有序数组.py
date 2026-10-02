class Solution(object):
    # 方法一：思路正确，但lc上要求是在nums1上改
    def merge_new(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i = 0
        j = 0
        nums3 = []
        while i<m and j<n:
            if nums1[i] <= nums2[j]:
                nums3.append(nums1[i])
                i += 1
            else:
                nums3.append(nums2[j])
                j += 1
        nums3.extend(nums1[i:m])
        nums3.extend(nums2[j:])
        for k in range(m + n):
            nums1[k] = nums3[k]
        


    # 方法二：从后往前填
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i = m-1
        j = n-1
        k = m+n-1
        while i>=0 and j>=0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        for i in range(j+1):
            nums1[i] = nums2[i]



for f in [Solution().merge_new, Solution().merge]:
    a = [1,2,3,0,0,0]
    f(a, 3, [2,5,6], 3)
    print(f.__name__, "->", a)


        
        