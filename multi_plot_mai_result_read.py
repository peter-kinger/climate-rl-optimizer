# -*- encoding: utf-8 -*-
'''
@File    :   multi_plot_mai_result_read.py
@Time    :   2025/04/30 17:11:37
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@Description :   这个版本的代码主要是读取运行完的结果来进行绘制高位空间图片的，依赖于嵌套训练的轨迹图
'''

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
        hairy_lines_path = 'data/without_rl_100'    
    if coord_cols is None:
        coord_cols = ('T_a', 'C_a', 'E21')

    if axes is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axes # 读取外部的绘制
        
    # 颜色的选择
    colortop = "lime"
    colorbottom = "black"

    # 循环读取 data\without_rl 里面的 csv 文件，并进行3维变量绘制
    for file_name in os.listdir(hairy_lines_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(hairy_lines_path, file_name)
            df = pd.read_csv(file_path)
            # 绘制3维变量
            ax.plot(df[coord_cols[0]], df[coord_cols[1]], df[coord_cols[2]], color=colorbottom if df['T_a'].iloc[-1] > 1.5 else colortop, linewidth=1, alpha=.08)

def plot_compare_lines(compare_lines_path=None, coord_cols=None, fig=None, axes=None):
    """比较没有 rl 管理的结果，轨迹将会是什么样的

    Args:
        compare_lines_path (_type_, optional): _description_. Defaults to None.
        coord_cols (_type_, optional): _description_. Defaults to None.
        fig (_type_, optional): _description_. Defaults to None.
        axes (_type_, optional): _description_. Defaults to None.
    """

    if compare_lines_path is None:
        compare_lines_path = 'data\iseec_run_data'   
         
    if coord_cols is None:
        coord_cols = ('T_a', 'C_a', 'E21')

    if axes is None:
        fig = plt.figure(figsize=(8,6)) 
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axes # 读取外部的绘制
        
    # 颜色的选择
    colortop = "lime"
    colorbottom = "black"

    # 循环读取 data\without_rl 里面的 csv 文件，并进行3维变量绘制
    for file_name in os.listdir(compare_lines_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(hairy_lines_path, file_name)
            
            # 利用 df 读取 xlsx 文件
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            
            # 只读取 time 在 2017到2099 的数据
            df = df[(df['time']>=2017) & (df['time']<=2099)]
            # 绘制3维变量
            ax.plot(df[coord_cols[0]], df[coord_cols[1]], df[coord_cols[2]], linewidth=1, alpha=.08)

    

def plot_colored_trajectory(csv_path, coord_cols=None, action_col=None, fig=None, axes=None,
                            cmap_name='viridis', linewidth=2):
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
    coords = df[list(coord_cols)].values    # shape=(N,3)
    actions = df.iloc[:, action_col].values   # shape=(N,)
    
    # 3) 归一化 action，用于 colormap
    # 这行代码创建了一个归一化器，将 actions 数组中的值映射到 [0,1] 区间
    norm = plt.Normalize(vmin=actions.min(), vmax=actions.max())
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(actions))
    
    # 4) 绘图
    if axes is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axes # 读取外部的绘制
    
    # 按线段逐条绘制，确保每段用该段起点的 action 着色
    for i in range(len(coords)-1):
        xs, ys, zs = coords[i:i+2, 0], coords[i:i+2, 1], coords[i:i+2, 2]
        ax.plot(xs, ys, zs, color=colors[i], linewidth=linewidth)
        
        # 增加 action label 部分
        if actions[i] != actions[i+1]:  # 只在动作发生变化时添加标签
            ax.text(xs[0], ys[0], zs[0], f'a={actions[i]}', 
                   fontsize=8, backgroundcolor='white')
    
    # 可选：点状展示（对轨迹进行说明）
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
               c=actions, cmap=cmap_name, norm=norm, s=20)
    
    # 对起点和终点作点状标记
    ax.scatter(coords[0,0], coords[0,1], coords[0,2], 
              color='red', s=100, marker='*', label='Start')
    ax.scatter(coords[-1,0], coords[-1,1], coords[-1,2], 
              color='green', s=100, marker='*', label='End')
    
    # # 添加起点终点的具体数值标注
    # ax.text(coords[0,0], coords[0,1], coords[0,2], 
    #         f'Start\n({coords[0,0]:.2f}, {coords[0,1]:.2f}, {coords[0,2]:.2f})', 
    #         fontsize=8, color='red')
    # ax.text(coords[-1,0], coords[-1,1], coords[-1,2], 
    #         f'End\n({coords[-1,0]:.2f}, {coords[-1,1]:.2f}, {coords[-1,2]:.2f})', 
    #         fontsize=8, color='green')
    
    # 首先绘制灰色半透明边界墙
    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    z_min, z_max = coords[:,2].min(), coords[:,2].max()
    
    # 创建三个面的网格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                        np.linspace(y_min, y_max, 20))
    xz, yz = np.meshgrid(np.linspace(x_min, x_max, 20),
                        np.linspace(z_min, z_max, 20))
    yz, zz = np.meshgrid(np.linspace(y_min, y_max, 20),
                        np.linspace(z_min, z_max, 20))
    
    # 绘制三个面
    ax.plot_surface(xx, yy, np.full_like(xx, z_min), alpha=0.1, color='gray')  # 底面
    ax.plot_surface(np.full_like(xz, x_max), yz, xz, alpha=0.1, color='gray')  # 右面
    ax.plot_surface(xz, np.full_like(xz, y_min), yz, alpha=0.1, color='gray')  # 左面
    
    # 增加对 action 的图例
    unique_actions = np.unique(actions)
    legend_elements = [plt.Line2D([0], [0], color=cmap(norm(action)), 
                                label=f'Action {action}', linewidth=2)
                      for action in unique_actions]
    
    # 修改这里的图例处理方式
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 5) 格式化
    ax.set_xlabel(coord_cols[0])  # 直接使用列名
    ax.set_ylabel(coord_cols[1])
    ax.set_zlabel(coord_cols[2])
    plt.title('3D Trajectory Colored by Action')
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                        ax=ax, pad=0.1)
    cbar.set_label('Action value')
    plt.tight_layout()
    
    plot_hairy_lines(fig=fig, axes=ax)
    # plot_compare_lines(fig=fig, axes=ax)
    
    plt.show()
    
    # TODO: 尾部增加多个 action 轨迹的相关说明
    
    return fig, ax

if __name__ == '__main__':
    # 举例：如果你的文件叫 trajectories.csv，
    # 前 3 列是 x,y,z，第 7 列是 action，就这样调用
    fig, ax3d = plot_colored_trajectory(f'output/sparse/rl_model_DQN_network_dict_pi_vf_default_800000/episode_0_results_20250430_113455.csv',coord_cols=('T_a', 'C_a', 'E21'), action_col=-3)
