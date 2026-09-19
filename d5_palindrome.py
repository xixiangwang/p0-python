# ===== d5_palindrome.py =====
# 本文件包含：① 回文判断（一行：s == s[::-1]）② 4 组测试含两个边界（"level"/"hello"/"a"/""）
# 相关：切片 [::-1] 的完整讲法 → d4_list.py 的「切片」段；字符串不可变 →《笔记》重点标记区
# ==============================

# 回文判断,判断一个字符串是不是回文
def is_palin(s):
    return s == s[::-1]
print(is_palin("level"))
print(is_palin("hello"))
print(is_palin("a"))
print(is_palin(""))

# 对于这个问题用python就非常方便，c语言的话还需要双指针