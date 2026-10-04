# =================================快速排序=================================
# 时间：O(n log n) · 最坏 O(n²)
# 空间：O(log n)（递归栈）
def quick_sort(nums):
    if len(nums) < 1:
        return nums
    pivot = nums[len(nums) // 2]

    left = [x for x in nums if x < pivot]
    mid = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]

    return quick_sort(left) + mid + quick_sort(right)

print(quick_sort([5,2,9,1,5,6]))





# =================================归并排序============================
# 时间：稳定 O(n log n)
# 空间：O(n)（要额外数组）
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])

    return merge(left,right)

def merge(a,b):
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

print(merge_sort([5,2,9,1,5,6]))