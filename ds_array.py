# 数组求最大值

def max_val(nums):
    if len(nums) == 0:
        return None

    max_v = nums[0]
    for n in nums:
        if n > max_v:
            max_v = n
    return max_v

print(max_val([3,5,1,7,9]))
print(max_val([]))