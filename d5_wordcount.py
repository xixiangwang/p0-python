# ===== d5_wordcount.py =====
# 本文件包含：① dict 字符计数 ② 绑小 list [次数, 字符] 让两者一起排序 ③ sort() + 负索引取 Top3
# 相关：计数公式 → d5_dict_str_set.py；为什么按次数排要绑小 list →《笔记》重点标记区 ⑪
# =============================

# 统计一段文本里每个字符出现的次数，然后输出出现最多的 3 个
def wordcount(s):
    d = {}
    for ch in s:                       # 第一步：dict统计出现次数
        d[ch] = d.get(ch,0) + 1

    pairs = []
    for ch in d:
        pairs.append([d[ch],ch])

    pairs.sort()

    for i in range(1,4):
        if i > len(pairs):
            break
        print(pairs[-i][0],pairs[-i][1])

    
wordcount("abcdaaadcbbbbc")
wordcount("aabb")
wordcount("aa")
wordcount("")