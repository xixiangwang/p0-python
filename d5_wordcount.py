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

    
print(wordcount("abcdaaadcbbbbc"))