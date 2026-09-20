# 冒泡排序
def bubble(nums):
    if len(nums) == 0:
        return []

    for j in range(0,len(nums)):
        for i in range(0,len(nums)-1):
            if nums[i] > nums[i+1]:
                nums[i],nums[i+1] = nums[i+1],nums[i]
    return nums

print(bubble([2,4,1,6,8,55]))
print(bubble([]))
