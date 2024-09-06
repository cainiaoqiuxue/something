import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/train_data.csv')
df = df[df['SoH'] != -100]
df['result'] = df['SoH']
sns.set_style('whitegrid')

# 创建主窗口
root = tk.Tk()
root.title("SoH交互图形绘制")

# 创建数据输入框
# data_entry = tk.Entry(root, width=50)
# data_entry.pack()

# 创建下拉菜单，用于选择绘制的图形类型
chart_type_var = tk.StringVar()
chart_type_dropdown = tk.OptionMenu(root, chart_type_var, "SoH容量图", "SoH分段图")
chart_type_dropdown.pack()

# 创建按钮，用于触发绘图动作
# draw_button = tk.Button(root, text="绘制图形", command=lambda: draw_chart(data_entry.get(), chart_type_var.get()))
draw_button = tk.Button(root, text="绘制图形", command=lambda: draw_chart('123', chart_type_var.get()))
draw_button.pack()

# 创建画布和子图
fig = Figure(figsize=(8, 6), dpi=100)
ax = fig.add_subplot(111)

# 初始化画布
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 定义绘制图形的函数
def draw_chart(data, chart_type):
    # 清空原有图像
    ax.clear()
    
    # 解析用户输入的数据
    try:
        data = list(map(float, data.split(",")))
    except ValueError:
        print("输入数据格式错误，请重新输入。")
        return
    
    # 根据选择的图形类型绘制图形
    if chart_type == "SoH容量图":
        # ax.plot(data, marker='o')
        sns.lineplot(df, x='result', y='capacity', hue='CS_Name', ax=ax)
    elif chart_type == "SoH分段图":
        # ax.bar(range(len(data)), data)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for i in range(len(bins) - 1):
            sns.lineplot(df[(df['result'] > bins[i]) & (df['result'] <= bins[i + 1])], x='result', y='capacity', ax=ax)
    # elif chart_type == "散点图":
    #     ax.scatter(range(len(data)), data)
    
    ax.set_xlabel('SoH')
    ax.set_ylabel('capacity')
    ax.set_title('SoH Plot')
    
    # 刷新画布显示
    canvas.draw()

# 运行主循环
root.mainloop()