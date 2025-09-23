import random
import os.path
import tqdm
import sys

# 配置参数
random.seed(1)

# 根据命令行参数确定机器人URDF路径
ROBOT_URDF_PATH = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/jaco_7/jaco_7s.urdf"  # 默认Jaco机器人
if len(sys.argv) > 1:
    ROBOT_URDF_PATH = sys.argv[1]  # 允许通过命令行指定URDF路径

print(f"使用机器人: {ROBOT_URDF_PATH}")


def get_robot_workspace_bounds(robot_urdf_path):
    """
    获取机器人工作空间边界

    Args:
        robot_urdf_path: 机器人URDF文件路径

    Returns:
        dict: 包含工作空间边界的字典
    """
    # 生成工作空间文件名
    robot_name = os.path.splitext(os.path.basename(robot_urdf_path))[0]
    workspace_file = f"{robot_name}_workspace.json"
    # 使用默认的保守估计
    workspace_bounds = {
        "x_start": -0.8,
        "x_end": 0.8,
        "y_start": -0.8,
        "y_end": 0.8,
        "z_start": 0.1,
        "z_end": 1.0,
    }

    if workspace_bounds is None:
        print(f"工作空间文件 {workspace_file} 不存在，正在分析工作空间...")

        # 动态导入并运行工作空间分析器
        try:
            from workspace_analyzer import WorkspaceAnalyzer

            analyzer = WorkspaceAnalyzer(robot_urdf_path)
            if analyzer.load_robot():
                positions = analyzer.sample_workspace(num_samples=1000)
                workspace_bounds = analyzer.analyze_workspace_bounds(positions)
                analyzer.save_workspace_bounds(workspace_bounds, workspace_file)
                print(f"工作空间分析完成，结果保存到 {workspace_file}")
            analyzer.disconnect()

        except Exception as e:
            print(f"工作空间分析失败: {e}")
            print("使用默认工作空间范围")

    return workspace_bounds


# 获取机器人工作空间边界
workspace_bounds = get_robot_workspace_bounds(ROBOT_URDF_PATH)

# 根据机器人工作空间设置场景生成参数
length = 0.07  # 体素大小
xstart = workspace_bounds["x_start"]
xend = workspace_bounds["x_end"]
ystart = workspace_bounds["y_start"]
yend = workspace_bounds["y_end"]
zstart = workspace_bounds["z_start"]
zend = workspace_bounds["z_end"]

print("工作空间范围:")
print(f"  X: [{xstart:.3f}, {xend:.3f}]")
print(f"  Y: [{ystart:.3f}, {yend:.3f}]")
print(f"  Z: [{zstart:.3f}, {zend:.3f}]")

xlist = []
t = xstart
while t < xend:
    xlist.append(t)
    t = t + length
# print(xlist)
ylist = []
t = ystart
while t < yend:
    ylist.append(t)
    t = t + length

zlist = []
t = zstart
while t < zend:
    zlist.append(t)
    t = t + length
# num_ob=sys.argv[1]


def find_nearest(x1, x2, xlist):
    """改进版本，使用二分查找提高效率"""
    import bisect

    # 使用二分查找找到插入位置
    lower = max(0, bisect.bisect_left(xlist, x1) - 1)
    upper = max(0, bisect.bisect_right(xlist, x2) - 1)

    return (lower, upper)


def find_collision(x1, y1, z1, x2, y2, z2):
    list_voxels = []
    # print(x1,y1,z1,x2,y2,z2)
    x1, x2 = find_nearest(x1, x2, xlist)
    y1, y2 = find_nearest(y1, y2, ylist)
    z1, z2 = find_nearest(z1, z2, zlist)
    # print(x1,y1,z1,x2,y2,z2)
    for i in range(z1, z2 + 1):
        for j in range(y1, y2 + 1):
            for k in range(x1, x2 + 1):
                list_voxels.append((k, j, i))
    return list_voxels


def remove_dup(list_voxels):
    new = []
    for i in list_voxels:
        if i not in new:
            new.append(i)
    return new


def write_mujoco_header(f, robot_urdf_path):
    """写入MuJoCo格式的文件头部"""
    f.write('<?xml version="1.0"?>\n')
    f.write('<mujoco model="obstacle_scene">\n')
    f.write('  <compiler angle="radian" coordinate="local"/>\n')
    f.write('  <option timestep="0.001" gravity="0 0 -9.81"/>\n')
    f.write("\n  <asset>\n")
    f.write('    <material name="obstacle_mat" rgba="0.8 0.8 0.0 0.8"/>\n')
    f.write('    <material name="ground_mat" rgba="0.5 0.5 0.5 1"/>\n')
    f.write("  </asset>\n\n")
    f.write("  <worldbody>\n")
    f.write("    <!-- Ground plane -->\n")
    f.write(
        '    <geom name="ground" type="plane" size="3 3 0.1" material="ground_mat"/>\n\n'
    )
    # f.write("    <!-- Robot -->\n")
    # f.write(f'    <body name="robot" file="{robot_urdf_path}"/>\n\n')
    f.write("    <!-- Obstacles -->\n")


def write_mujoco_obstacle(f, obstacle_id, scale, position, color):
    """写入单个MuJoCo格式的障碍物"""
    f.write(
        f'    <body name="obstacle_{obstacle_id}" pos="{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}">\n'
    )
    f.write(
        f'      <geom name="obs_{obstacle_id}" type="box" size="{scale[0] / 2:.6f} {scale[1] / 2:.6f} {scale[2] / 2:.6f}" rgba="{color} 0.8"/>\n'
    )
    f.write("    </body>\n")


def write_mujoco_footer(f):
    """写入MuJoCo格式的文件尾部"""
    f.write("  </worldbody>\n")
    f.write("</mujoco>\n")


# 定义一个机器人基座周围的避让区域（立方体），防止障碍物生成得太近
# 尺寸为 [min, max]，单位为米
KEEPOUT_BOX = {
    "x": [-0.25, 0.25],
    "y": [-0.25, 0.25],
    "z": [0.0, 0.3],  # 从地面到0.3米高
}
print("\n使用机器人基座的避让区域:")
print(f"  X: {KEEPOUT_BOX['x']}")
print(f"  Y: {KEEPOUT_BOX['y']}")
print(f"  Z: {KEEPOUT_BOX['z']}")

voxel_dict = {}
color = ["0.2 0.2 0.0", "0.5 0.5 0.0", "0.8 0.8 0.0"]
for num_ob in [3, 6, 9, 12]:
    for i1 in range(0, len(zlist)):
        for j in range(0, len(ylist)):
            for k in range(0, len(xlist)):
                voxel_dict[(k, j, i1)] = 0
    os.makedirs("scene_benchmarks/dens" + str(num_ob), exist_ok=True)
    # os.makedirs("voxel_object_collision/jaco/dens"+str(num_ob), exist_ok=True)
    # fvoxel=open("voxel_object_collision/jaco/dens"+str(num_ob)+"/summary.txt","w")
    print(num_ob)
    sum_voxels = 0
    for i in tqdm.tqdm(range(0, 100)):
        # num_ob = int(sys.argv[1])  # int(random.uniform(4,4))
        list_voxels = []
        # fv= open("voxel_object_collision/jaco/dens"+str(num_ob)+"/scene_"+str(i)+".txt","w")

        # MuJoCo格式文件
        f = open(
            "scene_benchmarks/dens" + str(num_ob) + "/obstacles_" + str(i) + ".xml",
            "w",
        )
        write_mujoco_header(f, ROBOT_URDF_PATH)

        # print(num_ob)
        objects = int(random.uniform(num_ob, num_ob + 2))
        for j in range(0, objects):
            # --- 修改: 循环生成障碍物，直到其不与基座避让区碰撞 ---
            max_retries = 100
            for _ in range(max_retries):
                # 调整障碍物大小使其适合机器人工作空间
                workspace_x_range = xend - xstart
                workspace_y_range = yend - ystart
                workspace_z_range = zend - zstart

                # 障碍物尺寸为工作空间的5%-20%
                xscale = random.uniform(
                    workspace_x_range * 0.05, workspace_x_range * 0.2
                )
                yscale = random.uniform(
                    workspace_y_range * 0.05, workspace_y_range * 0.2
                )
                zscale = random.uniform(
                    workspace_z_range * 0.05, workspace_z_range * 0.2
                )

                # 确保障碍物位置在工作空间内
                xpos = random.uniform(xstart + xscale / 2, xend - xscale / 2)
                ypos = random.uniform(ystart + yscale / 2, yend - yscale / 2)
                zpos = random.uniform(zstart + zscale / 2, zend - zscale / 2)

                # 检查障碍物是否与避让区重叠
                obs_xmin, obs_xmax = xpos - xscale / 2, xpos + xscale / 2
                obs_ymin, obs_ymax = ypos - yscale / 2, ypos + yscale / 2
                obs_zmin, obs_zmax = zpos - zscale / 2, zpos + zscale / 2

                # AABB (轴对齐包围盒) 重叠测试
                x_overlap = (
                    obs_xmin < KEEPOUT_BOX["x"][1] and obs_xmax > KEEPOUT_BOX["x"][0]
                )
                y_overlap = (
                    obs_ymin < KEEPOUT_BOX["y"][1] and obs_ymax > KEEPOUT_BOX["y"][0]
                )
                z_overlap = (
                    obs_zmin < KEEPOUT_BOX["z"][1] and obs_zmax > KEEPOUT_BOX["z"][0]
                )

                if not (x_overlap and y_overlap and z_overlap):
                    # 如果不重叠，则此障碍物位置有效，跳出重试循环
                    break
            else:
                # 如果循环完成（即达到最大重试次数）而没有break
                print(
                    f"\n警告: 场景 {i} 中, 无法为障碍物 {j} 找到避让区域外的位置 (已尝试 {max_retries} 次). 跳过此障碍物."
                )
                continue  # 跳到下一个障碍物j

            # --- 修改结束 ---

            # 写入有效的障碍物到MuJoCo文件
            write_mujoco_obstacle(
                f,
                j,
                [xscale, yscale, zscale],
                [xpos, ypos, zpos],
                color[(j) % 3],
            )

            temp = find_collision(
                xpos, ypos, zpos, xpos + xscale, ypos + yscale, zpos + zscale
            )
            list_voxels = list_voxels + temp

        uniq_voxels = remove_dup(list_voxels)
        sum_voxels += len(uniq_voxels)

        for v in uniq_voxels:
            voxel_dict[v] += 1
            # fv.write("%s\n"%(str(v)))

        write_mujoco_footer(f)
        f.close()
        # fv.close()
