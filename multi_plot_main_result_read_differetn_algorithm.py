# -*- encoding: utf-8 -*-
"""
@File    :   multi_plot_mai_result_read.py
@Time    :   2025/04/30 17:11:37
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@Description :   这个版本的代码主要是读取运行完的结果来进行绘制高位空间图片的，依赖于嵌套训练的轨迹图
"""

# here put the import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_hairy_lines(hairy_lines_path=None, coord_cols=None, fig=None, axes=None):
    """通过外部的数据来进行绘制轨迹

    Args:
        hairy_lines_path (_type_, optional): _description_. Defaults to None.
        coord_cols (_type_, optional): _description_. Defaults to None.
        fig (_type_, optional): _description_. Defaults to None.
        axes (_type_, optional): _description_. Defaults to None.
    """
    # 循环读取 data\without_rl 里面的 csv 文件，并进行3维变量绘制
    if hairy_lines_path is None:
        hairy_lines_path = "data/without_rl_100"
    if coord_cols is None:
        coord_cols = ("T_a", "C_a", "E21")

    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axes  # 读取外部的绘制

    # 颜色的选择
    colortop = "lime"
    colorbottom = "black"

    # 循环读取 data\without_rl 里面的 csv 文件，并进行3维变量绘制
    for file_name in os.listdir(hairy_lines_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(hairy_lines_path, file_name)
            df = pd.read_csv(file_path)
            # 绘制3维变量
            ax.plot(
                df[coord_cols[0]],
                df[coord_cols[1]],
                df[coord_cols[2]],
                color=colorbottom if df["T_a"].iloc[-1] > 1.5 else colortop,
                linewidth=1,
                alpha=0.08,
            )


def plot_compare_lines(compare_lines_path=None, coord_cols=None, fig=None, axes=None):
    """比较没有 rl 管理的结果，轨迹将会是什么样的
    （即 rl 管理中的几个经典 case）

    Args:
        compare_lines_path (_type_, optional): _description_. Defaults to None.
        coord_cols (_type_, optional): _description_. Defaults to None.
        fig (_type_, optional): _description_. Defaults to None.
        axes (_type_, optional): _description_. Defaults to None.
    """

    if compare_lines_path is None:
        compare_lines_path = "data\iseec_run_data"

    if coord_cols is None:
        coord_cols = ("T_a", "C_a", "E21")

    if axes is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axes  # 读取外部的绘制

    # 颜色的选择
    colortop = "lime"
    colorbottom = "black"

    # 循环读取 data\without_rl 里面的 csv 文件，并进行3维变量绘制
    for file_name in os.listdir(compare_lines_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(hairy_lines_path, file_name)

            # 利用 df 读取 xlsx 文件
            df = pd.read_excel(file_path, sheet_name="Sheet1")

            # 只读取 time 在 2017到2099 的数据
            df = df[(df["time"] >= 2017) & (df["time"] <= 2099)]
            # 绘制3维变量
            ax.plot(
                df[coord_cols[0]],
                df[coord_cols[1]],
                df[coord_cols[2]],
                linewidth=1,
                alpha=0.08,
            )


def plot_colored_trajectory(
    csv_path, coord_cols=None, action_col=None, fig=None, axes=None, colour=None
):
    """
    读取 csv，绘制 3D 轨迹，并根据 action 着色。

    Parameters
    ----------
    csv_path : str
        CSV 文件路径。
    coord_cols : tuple of int
        表示 (x, y, z) 的列索引。
    action_col : int
        动作所在列的索引。
    cmap_name : str
        matplotlib colormap 名称，用于 action 到颜色的映射。
    linewidth : float
        轨迹线段宽度。
    """
    # 1) 读取
    df = pd.read_csv(csv_path)

    # 2) 坐标与动作
    coords = df[list(coord_cols)].values  # shape=(N,3)
    actions = df.iloc[:, action_col].values  # shape=(N,)

    # 3) 归一化 action，用于 colormap
    # 这行代码创建了一个归一化器，将 actions 数组中的值映射到 [0,1] 区间
    norm = plt.Normalize(vmin=actions.min(), vmax=actions.max())
    cmap = plt.get_cmap("Set1")  # 或者使用 tab10

    colors = cmap(norm(actions))  # 直接是根据颜色的值来进行归一化

    # 4) 绘图
    if axes is None and colour == None:  # 单一算法的绘制
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 按线段逐条绘制，确保每段用该段起点的 action 着色
        for i in range(len(coords) - 1):
            xs, ys, zs = (
                coords[i : i + 2, 0],
                coords[i : i + 2, 1],
                coords[i : i + 2, 2],
            )
            ax.plot(xs, ys, zs, color=colors[i], linewidth=4)

            # # 增加 action label 部分
            # if actions[i] != actions[i+1]:  # 只在动作发生变化时添加标签
            #     ax.text(xs[0], ys[0], zs[0], f'a={actions[i]}',
            #            fontsize=8, backgroundcolor='white')

            # # 可选：动作变化点加标注
            # if i == 0 or actions[i] != actions[i-1]:
            #     ax.text(xs[0], ys[0], zs[0], f'{actions[i]}', fontsize=8, color=colors[i])
    elif axes is None and colour is not None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 按线段逐条绘制，确保每段用该段起点的 action 着色
        for i in range(len(coords) - 1):
            xs, ys, zs = (
                coords[i : i + 2, 0],
                coords[i : i + 2, 1],
                coords[i : i + 2, 2],
            )
            ax.plot(xs, ys, zs, color=colour, linewidth=4)

            # # 增加 action label 部分
            # if actions[i] != actions[i+1]:  # 只在动作发生变化时添加标签
            #     ax.text(xs[0], ys[0], zs[0], f'a={actions[i]}',
            #            fontsize=8, backgroundcolor='white')

            # # 可选：动作变化点加标注
            # if i == 0 or actions[i] != actions[i-1]:
            #     ax.text(xs[0], ys[0], zs[0], f'{actions[i]}', fontsize=8, color=colors[i])

    else:  # 多个算法的绘制
        ax = axes  # 读取外部的绘制, 多个算法的绘制
        # 按线段逐条绘制，确保每段用该段起点的 action 着色
        for i in range(len(coords) - 1):
            xs, ys, zs = (
                coords[i : i + 2, 0],
                coords[i : i + 2, 1],
                coords[i : i + 2, 2],
            )
            ax.plot(xs, ys, zs, color=colour, linewidth=4)

    # 对起点和终点作点状标记
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        coords[0, 2],
        color="red",
        s=50,
        marker="o",
        label="Start",
    )
    ax.scatter(
        coords[-1, 0],
        coords[-1, 1],
        coords[-1, 2],
        color="green",
        s=50,
        marker="o",
        label="End",
    )  # maker 表示具体的标记样式

    # # 添加起点终点的具体数值标注
    # ax.text(coords[0,0], coords[0,1], coords[0,2],
    #         f'Start\n({coords[0,0]:.2f}, {coords[0,1]:.2f}, {coords[0,2]:.2f})',
    #         fontsize=8, color='red')
    # ax.text(coords[-1,0], coords[-1,1], coords[-1,2],
    #         f'End\n({coords[-1,0]:.2f}, {coords[-1,1]:.2f}, {coords[-1,2]:.2f})',
    #         fontsize=8, color='green')

    # 画 C_a=945 的墙（Y轴墙）
    Y_wall = 945
    X_min, X_max = ax.get_xlim()
    Z_min, Z_max = ax.get_zlim()
    X = np.linspace(X_min, 1.5, 20)  # 右边界正好到1.5
    Z = np.linspace(Z_min, Z_max, 20)
    X, Z = np.meshgrid(X, Z)
    Y = np.full_like(X, Y_wall)
    ax.plot_surface(X, Y, Z, color="gray", alpha=0.1)

    # 画 T_a=1.5 的墙（X轴墙）
    X_wall = 1.5
    Y_min, Y_max = ax.get_ylim()
    Z2 = np.linspace(Z_min, Z_max, 20)
    Y2 = np.linspace(Y_min, 945, 20)  # 上边界正好到945
    Y2, Z2 = np.meshgrid(Y2, Z2)
    X2 = np.full_like(Y2, X_wall)
    ax.plot_surface(X2, Y2, Z2, color="gray", alpha=0.1)

    # # 增加一个 T_a = 2 的墙
    # X_wall_2 = 2.0
    # Y_min, Y_max = ax.get_ylim()
    # Z2 = np.linspace(Z_min, Z_max, 20)
    # Y2 = np.linspace(Y_min, 945, 20)  # 上边界正好到945
    # Y2, Z2 = np.meshgrid(Y2, Z2)
    # X2_2 = np.full_like(Y2, X_wall_2)
    # ax.plot_surface(X2_2, Y2, Z2, color='gray', alpha=0.1)

    # # 画 E21 的墙（Z轴墙）
    # Z_wall = 25
    # X_min, X_max = ax.get_xlim()
    # Y_min, Y_max = ax.get_ylim()
    # X3 = np.linspace(X_min, X_max, 20)
    # Y3 = np.linspace(Y_min, Y_max, 20)
    # X3, Y3 = np.meshgrid(X3, Y3)
    # Z3 = np.full_like(X3, Z_wall)
    # ax.plot_surface(X3, Y3, Z3, color='gray', alpha=0.1)

    # 增加对 action 的图例
    unique_actions = np.unique(actions)
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=cmap(norm(action)),  # 说明颜色部分线图例
            label=f"Action {action}",
            linewidth=2,
        )
        for action in unique_actions
    ]

    # 轨迹颜色与算法的图例
    color_desc = {
        # "green": "multi-objective reward type 1",
        "pink": "DQN algorithm",
        "blue": "PPO algorithm",
        "red": "A2C algorithm",
    }
    color_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=desc)
        for color, desc in color_desc.items()
    ]
    # 起点终点的图例
    point_handles = [
        plt.Line2D(
            [0], [0], color="red", marker="o", linestyle="", markersize=8, label="start"
        ),
        plt.Line2D(
            [0], [0], color="green", marker="o", linestyle="", markersize=8, label="end"
        ),
    ]

    # 合并所有图例元素
    all_handles = color_handles + point_handles  # color_handles + point_handles + legend_elements

    # 修改这里的图例处理方式
    ax.legend(handles=all_handles, loc='upper right')

    # 5) 格式化
    ax.set_xlabel("atmospheric temperature" + " " + coord_cols[0] + "[℃]")  # 直接使用列名
    ax.set_ylabel("atmospheric concentration" + " " + coord_cols[1] + "[gtc]")
    ax.set_zlabel("existing renewable energy" + " " + r"$E_{21}$" + "[EJ]")

    plt.title("3D Trajectory Colored by Action")

    # 绘制 action 值的颜色条
    # cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap) # 这里使用 ScalarMappable 来创建颜色条
    #                     ax=ax, pad=0.1)
    # cbar.set_label('Action value')

    plt.tight_layout()  # Matplotlib 的一个常用函数，用于自动调整子图参数

    # optional: 算法对应颜色说明，手动对颜色算法进行标注

    # optional: 右侧单独的图例说明
    # 增加动作意义的单独说明
    # action_desc = {  # 只说明加快的部分
    #     2: "Accelerate the development of renewable energy",
    #     4: "Speed the progress of ACE",
    #     6: "renewable energy + ACE",
    #     12: "ACE + Accelerate the investment of renewable Energy",
    #     15: "all actions",
    # }
    # desc_lines = [f"{k}: {v}" for k, v in action_desc.items()]
    # desc_text = "\n".join(desc_lines)
    # plt.subplots_adjust(right=0.75)  # 给右侧留空间
    # fig.text(
    #     0.78,
    #     0.5,
    #     desc_text,
    #     va="center",
    #     ha="left",
    #     fontsize=12,
    #     bbox=dict(facecolor="white", edgecolor="black"),
    # )

    # optinal: 在基础的轨迹绘制上增加其他部分的绘制，启用方式：直接取消注释即可
    # plot_hairy_lines(fig=fig, axes=ax)
    # plot_compare_lines(fig=fig, axes=ax)

    ax.grid(False)  # 取消网格显示
    # plt.show() #

    return fig, ax


if __name__ == "__main__":
    # 举例：如果你的文件叫 trajectories.csv，
    # 前 3 列是 x,y,z，第 7 列是 action，就这样调用

    # optinal： 绘制单个算法情况
    # fig, ax3d = plot_colored_trajectory(f'supplyment/save_future_data/exp122_weights82_DQN_episode_1_results_20250619_204927.csv',coord_cols=('T_a', 'C_a', 'E21'), action_col=-3)

    # # # optinal： 绘制多个算法情况
    fig, ax3d = plot_colored_trajectory(
        f"supplyment/save_future_data/PPO_exp822_episode_1_results_20250624_130703.csv",
        coord_cols=("T_a", "C_a", "E21"),
        action_col=-3,
        colour="blue",
    )
    
    # plot_colored_trajectory(
    #     f"supplyment/save_future_data/exp122_weights82_DQN_episode_1_results_20250619_204927.csv",
    #     coord_cols=("T_a", "C_a", "E21"),
    #     action_col=-3,
    #     fig=fig,
    #     axes=ax3d,
    #     colour="green",
    # )

    # # 不同的 reward 结果绘制
    plot_colored_trajectory(
        f"supplyment/save_future_data/exp822_DQN_episode_1_results_20250606_142557.csv",
        coord_cols=("T_a", "C_a", "E21"),
        action_col=-3,
        fig=fig,
        axes=ax3d,
        colour="pink",
    )

    # # # 不同算法继续在一个图坐标上绘制
    plot_colored_trajectory(f'supplyment/save_future_data/exp822_A2C_episode_1_results_20250606_153959.csv', coord_cols=('T_a', 'C_a', 'E21'), action_col=-3,fig=fig,axes=ax3d,colour='red')

    # plot_colored_trajectory(f'supplyment/save_future_data/exp1013_DQN_episode_1_results_20250606_161916.csv', coord_cols=('T_a', 'C_a', 'E21'), action_col=-3,fig=fig,axes=ax3d,colour='orange')

    # plot_colored_trajectory(f"output/multi_objective_single_T_a_exp8/rl_model_fixed_action_hariy_network_Netxxx_no_debug_plot_100/episode_0_results_20250527_195522.csv", coord_cols=('T_a', 'C_a', 'E21'), action_col=-3,fig=fig,axes=ax3d,colour='red')

    # plot_colored_trajectory(f"output/multi_objective_governance_social_foundations_exp5/rl_model_DQN_network_dict_pi_vf_default_600000/episode_0_results_20250520_212556.csv", coord_cols=('T_a', 'C_a', 'E21'), action_col=-3,fig=fig,axes=ax3d,colour='orange')
    
    plt.show()


