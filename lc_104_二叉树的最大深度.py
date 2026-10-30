# 递归,自下而上可以，应该还是在遍历的基础上
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root:
            return 0      # 这里就会有深度了，切入口
        
        return  max(self.maxDepth(root.left),self.maxDepth(root.right)) + 1


# 注意self.