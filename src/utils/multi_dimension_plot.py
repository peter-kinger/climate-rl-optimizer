
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import odeint

def plot_hairy_lines(num, ax3d, env):
    """绘制毛线图
    """
    colortop = "lime"
    colorbottom = "black"
    iseec_0 = np.random.rand(num, 10) # TODO，初始扰动状态的部分
    time = np.linspace(0, 81, 1000)
    

  
    for i in range(num):
        x0 = iseec_0[i]
        traj = odeint(env.iseec_dynamics_v1_ste, x0, time, mxstep=50000)
        
        # 一条一条绘制
        ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                    color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.08)
    
    


def create_figure(Azimut=170, Elevation=25, label=None, colors=None, ax=None, ticks=True, plot_boundary=True,):
    """创建基础画图要素包含标签等（其他图都是基于此进行绘制）
    """
    if ax is None:
        fig3d = plt.figure(figsize=(9,9))
        #ax3d = plt3d.Axes3D(fig3d)
        ax3d = fig3d.add_subplot(111, projection="3d")
    # else:
    #     ax3d=ax
    #     fig3d=None

    # if ticks==True:
    #     make_3d_ticks(ax3d)
    # else:
    #     ax3d.set_xticks([])
    #     ax3d.set_yticks([])
    #     ax3d.set_zticks([])
    
    # A_PB=[10, 265]
    # top_view=[25,170]
    # AZIMUTH, ELEVATION =  Azimut,Elevation
    # ax3d.view_init(ELEVATION, AZIMUTH)
    
    
    S_scale = 1e9
    Y_scale = 1e12
    ax3d.set_xlabel("\n\nexcess atmospheric carbon\nstock A [GtC]", )
    ax3d.set_ylabel("\n\neconomic output Y \n  [%1.0e USD/yr]"%Y_scale, )
    ax3d.set_zlabel("\n\nrenewable knowledge\nstock S [%1.0e GJ]"%S_scale,)

    # # Add boundaries to plot
    # if plot_boundary:
    #     ays_general.add_boundary(ax3d,
    #                              sunny_boundaries=["planetary-boundary", "social-foundation"],
    #                              **ays.grid_parameters, **ays.boundary_parameters)

    ax3d.grid(False)

    # legend_elements = [] 
    # if label is None:
    #     # For Management Options
    #     for idx in range(len(management_options)):
    #             legend_elements.append(Line2D([0], [0], lw=2, color=color_list[idx], label=management_options[idx]))
        
    #     #ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=shelter_color, label='Shelter')
    # else:
    #     for i in range(len(label)):
    #         ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color=colors[i], label=label[i])

    # For Startpoint
    # ax3d.scatter(*zip([0.5,0.5,0.5]), lw=4, color='black')
    
    # For legend
    # legend_elements.append(Line2D([0], [0], lw=2, label='current state',  marker='o', color='w', markerfacecolor='red', markersize=15))   
    # ax3d.legend(handles=legend_elements,prop={'size': 14}, bbox_to_anchor=(0.85,.90), fontsize=20,fancybox=True, shadow=True)

    return fig3d, ax3d

def plot_run(learning_progress, env, fig, axes, colour, fname=None):
    """完成测试时候 agent 的轨迹图绘制

    Args:
        model (_type_): _description_
        env (_type_): _description_
    """
    intSteps = 2
    sim_time_step = np.linspace(0, 1, intSteps)
    
    if axes is None:
        fig, ax3d = create_figure() 
    else:
        ax3d = axes # 表示其他继续绘图的使用同一坐标轴
    
    for state_action in learning_progress:
        state = state_action[0]
        action = state_action[1]

        traj_one_step = odeint(env.iseec_dynamics_v1_ste, state, sim_time_step, mxstep=50000)
        color_list = ['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']
        my_color = color_list[action]
        # 绘制不同颜色的 action 的 trajectory
        ax3d.plot3D(xs=traj_one_step[:, 0], ys=traj_one_step[:, 1], zs=traj_one_step[:, 2],
                    color=my_color, alpha=0.3, lw=3)
        # 绘制不同算法的 trjectory
        ax3d.plot3D(xs=traj_one_step[:, 0], ys=traj_one_step[:, 1], zs=traj_one_step[:, 2],
                    color=colour, alpha=0.3, lw=3)
        
    plot_hairy_lines(20, ax3d)
    
    # 保存结果在 plots 文件夹里
    if fname is not None:
        plt.savefig(fname)
    plt.show() 
    
    return fig, ax3d

def plot_current_state_trajectories(start_state=None, model=None, env=None, colour=None, steps=81, fname=None, ax3d=None, fig=None):
    """完成测试时候 agent 的轨迹图绘制的前期数据收集

    Args:
        start_state (_type_): _description_
        model (_type_): _description_
        env (_type_): _description_
        steps (int, optional): _description_. Defaults to 81.
        ax3d (_type_, optional): _description_. Defaults to None.
        fig (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    obs, _ = env.reset(start_state) # 相当于单个 episode 
    learning_progress = []
    actions = []
    rewards = []
    for step in range(steps):
        list_state = env.state # 对应的是没有执行 Action 的初始状态
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        learning_progress.append([list_state, action, reward])
        
        if done:
            break
    
    fig, ax3d = plot_run(learning_progress, env=env, fig=fig, axes=ax3d, fname=fname,colour=colour)
    
    return fig, ax3d




