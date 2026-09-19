# # dict字典
# d = {}                  # 建空字典（⚠️ {} 是 dict，不是 set）
# d["a"] = 1              # 存：键 -> 值
# d["a"]                  # 取：键不存在会 KeyError
# d.get("a")              # 安全取：不存在会返回 None
# d.get("a",0)            # 安全取：不存在会返回0
# "a" in d                # 键在不在字典里
# len(d)                  # 有几个键
# del d["a"]              # 删
# for k,v in d.items():   # 遍历：同时拿到键和值
#     print(k,v)

# # 核心：计数公式
# # 取出 ch 现在的次数（没有就是 0），加一，存回去。​ 这就是"计数"的全部
# count = {}
# for ch in "anagram":
#     count[ch] = count.get(ch, 0) + 1
# print(count)



# # str 方法 + set
# s = " Hello,World "                        # 注意这里是字符串
# print(s.strip())                           # 去两头空格
# print(s.strip().lower())                   # 去两头空格后又全转小写
# print(s.split(","))                        # 按逗号切开 -> 得到一个新List
# print("-".join(["a","b","c"]))             # 用"-"把List粘成字符串
# print(s.replace("World","Python"))         # 替换
# print(s.find("World"))                     # 找到的位置，没找到就返回-1
# print(s.startswith(" H"))                  # 是否以...开头


# d = {"a":1,"b":2}
# print(list(d.keys()))                           # List出所有键
# print(list(d.values()))                         # List出所有值
# del d["a"]                                      # 删掉一个键
# print(d)

# print({"outer":{"inner":10}}["outer"]["inner"]) # 嵌套dict，最外层的 dict 只有一个键 "outer"，它的"值"又是另一个 dict
# # 等价于
# d = {"outer": {"inner": 10}}
# step1 = d["outer"]          # 先拿到里面那个字典 → {'inner': 10}
# print(step1["inner"])       # 再从它里面取 → 10


a = {1,2,3}
b = {2,3,4}
print(a & b)
print(a | b)
print(a - b)
print(list({1,1,2,3,3}))