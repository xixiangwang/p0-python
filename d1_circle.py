# ===== d1_circle.py =====
# 本文件包含：① 输入半径 ② 算圆面积 ③ f-string 保留 2 位小数 {area:.2f}
# 相关：更基础的 f-string → d1_hello.py
# ==========================

r = float(input("请输入半径："))
area = 3.14159 * r * r
print(f"圆面积 = {area:.2f}")