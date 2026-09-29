# =================================二叉树=====================================

# 1. 节点类 —— 比链表多一个指针
class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right



# 2. 手工搭一棵树
root = TreeNode(1,TreeNode(2,TreeNode(4),TreeNode(5)),TreeNode(3))



# 3. 三种遍历 
def preorder(t):           # 前序：根 → 左 → 右
    if not t:
        return []
    return [t.val] + preorder(t.left) + preorder(t.right)
print("前序=",preorder(root))     # 前序 = [1, 2, 4, 5, 3]

def inorder(t):                   # 中序：左 → 根 → 右
    if not t:
        return []
    return inorder(t.left) + [t.val] + inorder(t.right)
print("中序=",inorder(root))      # 中序 = [4, 2, 5, 1, 3]

def postorder(t):                 # 后序：左 → 右 → 根
    if not t:
        return []
    return postorder(t.left) + postorder(t.right) + [t.val]
print("后序=",postorder(root))    # 后序= [4, 5, 2, 3, 1]



# 4. 中序的迭代版 —— 正好用你刚学的栈(有点难理解)
def inorder_iter(t):
    res = []
    st = []
    cur = t
    while cur or st:
        while cur:                      # 一路向左，全压栈
            st.append(cur)
            cur = cur.left
        node = st.pop()                 # 弹出来 = 最左那个还没访问的
        res.append(node.val)
        cur = node.right                # 转向右子树
    return res
print("中序=",inorder_iter(root))



             