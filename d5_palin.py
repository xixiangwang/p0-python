# 回文判断,判断一个字符串是不是回文
def is_palin(s):
    return s == s[::-1]
print(is_palin("level"))
print(is_palin("hello"))
print(is_palin("a"))
print(is_palin(""))

# 对于这个问题用python就非常方便，c语言的话还需要双指针