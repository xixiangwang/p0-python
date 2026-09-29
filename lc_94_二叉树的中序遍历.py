# ===================================递归版本==================================
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        if not root:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)





# =======================================迭代版本===================================
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

    def inorderTraversal_iter(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        res = []
        st = []
        cur = root
        while cur or st:
            while cur:
                st.append(cur)
                cur = cur.left
            node = st.pop()
            res.append(node.val)
            cur = node.right
        return res


root = TreeNode(1, None, TreeNode(2, TreeNode(3), None))
print(Solution().inorderTraversal(root))          # [1, 3, 2]
print(Solution().inorderTraversal_iter(root))     # [1, 3, 2]