import random
import os.path
import tqdm
import sys

# 配置参数
random.seed(1)
ROBOT_URDF_PATH = "/home/lanh/project/robot_sim/coll_prediction_artifact/data/robots/panda/panda.urdf"  # 默认机器人URDF路径
if len(sys.argv) > 1:
    ROBOT_URDF_PATH = sys.argv[1]  # 允许通过命令行指定URDF路径

length = 0.07
xstart = -1.12
xend = 1.12
ystart = -1.12
yend = 1.12
zstart = -1.12
zend = 1.12

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
    for i in range(0, len(xlist)):
        if x1 < xlist[i]:
            break
    lower = i - 1
    for i in range(0, len(xlist)):
        if x2 <= xlist[i]:
            break
    upper = i - 1
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
            xscale = random.uniform(length * 0, length * 8)
            xpos = random.uniform(xstart, xend)
            yscale = random.uniform(length * 0, length * 8)
            ypos = random.uniform(ystart, yend)
            zscale = random.uniform(length * 0, length * 8)
            zpos = random.uniform(zstart, zend)

            # 写入MuJoCo格式
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
