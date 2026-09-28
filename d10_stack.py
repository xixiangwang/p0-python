# ===================================栈（Stack）—— 后进先出 LIFO===========================

# 只能从顶上放、从顶上拿。最后放上去的最先被拿走
# Python 里不需要额外的东西 —— list 天然就是栈
# append(x) = 压栈；pop() 不带参数 = 弹栈顶

st = []
st.append(1)             # 进栈（压栈）
st.append(2)
st.append(3)
print(st)                # [1, 2, 3]

print(st.pop())          # 3   ← pop()会返回栈顶的值
print(st.pop())          # 2
print(st.pop())          # 1
print(st)                # []



# 验收是「4 个测试用例全部打印 PASS」,跑出来 4 行都是 PASS 就算过
def check(name,got,want):
    if got == want:
        print("PASS")
    else:
        print("FAIL")

st = []
st.append(1);st.append(2);st.append(3)
check("① 进 1,2,3 后长度",len(st),3)
check("② 栈顶是 3",st[-1],3)
check("③ pop 弹的是栈顶",st.pop(),3)
check("④ 弹完剩 [1,2]",st,[1,2])

