# type: ignore
"""
OBB Calculator - 机器人连杆有向包围盒计算模块

该模块使用CoACD和Open3D库为机器人连杆计算精确的有向包围盒(OBB)。

主要功能:
1. 使用CoACD进行凸分解，将复杂网格分解为多个凸包
2. 使用Open3D计算每个凸包的最小有向包围盒
3. 选择体积最小的OBB作为最终结果

核心函数:
- calculate_link_obbs(): 计算机器人所有连杆的OBB
- calculate_single_mesh_obb(): 计算单个网格的OBB
- analyze_mesh_complexity(): 分析网格复杂度

辅助函数:
- check_dependencies(): 检查依赖库是否可用
- get_default_coacd_params(): 获取默认CoACD参数

依赖库:
- yourdfpy: URDF文件解析
- trimesh: 3D网格处理
- open3d: 几何计算和OBB计算
- coacd: 凸近似凸分解

安装命令:
pip install yourdfpy trimesh open3d coacd

作者: VAMP项目组
版本: 1.0.0
"""

import numpy as np
from pathlib import Path
import tempfile
import os

# 检查必需的库
try:
    import yourdfpy
    import trimesh
    import open3d as o3d
    import coacd

    HAS_ALL_LIBS = True
    MISSING_LIBS = []
except ImportError:
    HAS_ALL_LIBS = False
    MISSING_LIBS = []

    # 检查各个库的可用性
    try:
        import yourdfpy
    except ImportError:
        MISSING_LIBS.append("yourdfpy")

    try:
        import trimesh
    except ImportError:
        MISSING_LIBS.append("trimesh")

    try:
        import open3d as o3d
    except ImportError:
        MISSING_LIBS.append("open3d")

    try:
        import coacd
    except ImportError:
        MISSING_LIBS.append("coacd")


def check_dependencies():
    """
    检查所有必需的依赖库是否已安装

    Returns:
        tuple: (is_available: bool, missing_libs: list)
    """
    return HAS_ALL_LIBS, MISSING_LIBS


def get_default_coacd_params():
    """
    获取CoACD的默认参数配置

    Returns:
        dict: CoACD参数字典
    """
    return {
        "threshold": 0.05,  # 凸性阈值，越小分解越精细
        "max_convex_hull": 32,  # 最大凸包数量
        "preprocess_mode": "auto",  # 预处理模式: 'auto', 'on', 'off'
        "preprocess_resolution": 50,  # 预处理分辨率
        "resolution": 2000,  # 采样分辨率
        "mcts_nodes": 20,  # MCTS节点数
        "mcts_iterations": 150,  # MCTS迭代数
        "mcts_max_depth": 3,  # MCTS最大深度
        "pca": False,  # 是否使用PCA
        "merge": True,  # 是否合并相似凸包
        "decimate": False,  # 是否简化网格
        "max_ch_vertex": 256,  # 每个凸包的最大顶点数
        "extrude": False,  # 是否挤出处理
        "extrude_margin": 0.01,  # 挤出边距
        "apx_mode": "ch",  # 近似模式: 'ch' (convex hull)
        "seed": 0,  # 随机种子
    }


def analyze_mesh_complexity(mesh, verbose=True):
    """
    分析网格的复杂度和几何特征

    Args:
        mesh: trimesh对象
        verbose (bool): 是否输出详细信息

    Returns:
        dict: 网格分析结果
    """
    if mesh.is_empty:
        return {"is_empty": True}

    # 基本统计信息
    analysis = {
        "is_empty": False,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "volume": mesh.volume if mesh.is_volume else 0,
        "surface_area": mesh.area,
    }

    # 几何复杂度分析
    try:
        # 计算凸性度量
        convex_hull = mesh.convex_hull
        analysis["convexity_ratio"] = (
            mesh.volume / convex_hull.volume if convex_hull.volume > 0 else 0
        )
        analysis["convex_hull_vertices"] = len(convex_hull.vertices)
        analysis["convex_hull_faces"] = len(convex_hull.faces)

        # 分析几何特征
        bounds = mesh.bounds
        analysis["bounding_box_volume"] = np.prod(bounds[1] - bounds[0])
        analysis["bbox_to_volume_ratio"] = (
            analysis["bounding_box_volume"] / analysis["volume"]
            if analysis["volume"] > 0
            else float("inf")
        )

        # 分析网格质量
        analysis["has_duplicate_faces"] = len(mesh.faces) != len(
            np.unique(mesh.faces, axis=0)
        )
        analysis["has_degenerate_faces"] = (mesh.face_areas < 1e-12).any()

    except Exception as e:
        analysis["analysis_error"] = str(e)

    if verbose:
        print("  网格分析结果:")
        print(f"    顶点数: {analysis['vertex_count']}, 面数: {analysis['face_count']}")
        print(
            f"    水密性: {analysis['is_watertight']}, 绕向一致: {analysis['is_winding_consistent']}"
        )
        print(f"    凸性比率: {analysis.get('convexity_ratio', 'N/A'):.4f}")
        print(f"    包围盒体积比: {analysis.get('bbox_to_volume_ratio', 'N/A'):.4f}")
        if analysis.get("convexity_ratio", 1.0) > 0.95:
            print("    ⚠️  网格已经非常接近凸形状，跳过凸分解")

    return analysis


def _load_robot_from_urdf(robot_urdf_path, verbose=True):
    """
    加载URDF文件并返回robot对象

    处理URDF文件中的路径问题：
    1. package://路径替换为绝对路径
    2. 相对路径（如meshes/finger.dae）前面添加URDF文件所在目录

    Args:
        robot_urdf_path (str): URDF文件路径
        verbose (bool): 是否输出详细信息

    Returns:
        tuple: (robot, temp_urdf_path) 其中temp_urdf_path需要在使用完毕后删除
    """
    import re

    urdf_dir = Path(robot_urdf_path).resolve().parent

    # 读取并修改URDF内容以替换各种路径
    with open(robot_urdf_path, "r") as f:
        urdf_content = f.read()

    # 1. 替换package://路径为绝对路径
    modified_content = urdf_content.replace("package://", str(urdf_dir) + "/")

    # 2. 处理相对路径：找到filename属性中的相对路径并转换为绝对路径
    # 匹配形如 filename="meshes/xxx.dae" 的模式
    def replace_relative_paths(match):
        filename = match.group(1)
        # 如果不是以/开头且不包含://，则认为是相对路径
        if not filename.startswith("/") and "://" not in filename:
            # 转换为绝对路径
            absolute_path = str(urdf_dir / filename)
            return f'filename="{absolute_path}"'
        else:
            # 已经是绝对路径或URL，保持不变
            return match.group(0)

    # 使用正则表达式替换相对路径
    modified_content = re.sub(
        r'filename="([^"]+)"', replace_relative_paths, modified_content
    )

    # 写入临时文件
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False
    ) as tmp_file:
        tmp_file.write(modified_content)
        temp_urdf_path = tmp_file.name

    # 使用yourdfpy库加载并解析URDF文件
    robot = yourdfpy.URDF.load(temp_urdf_path)

    return robot, temp_urdf_path


def _load_mesh_from_collision(collision, verbose=True):
    """
    从碰撞几何体中加载网格

    Args:
        collision: 碰撞几何体对象
        verbose (bool): 是否输出详细信息

    Returns:
        trimesh.Trimesh or None: 加载的网格对象，失败时返回None
    """
    if not (hasattr(collision.geometry, "mesh") and collision.geometry.mesh):
        return None

    mesh_path = collision.geometry.mesh.filename

    # 验证网格文件是否存在
    if not Path(mesh_path).exists():
        if verbose:
            print(f"  Mesh file not found: {mesh_path}")
        return None

    try:
        # 使用trimesh库加载3D网格文件
        loaded_object = trimesh.load(str(mesh_path))

        # 处理不同的返回类型
        if isinstance(loaded_object, trimesh.Scene):
            # 如果返回Scene对象，需要提取其中的网格
            if loaded_object.geometry:
                # 如果Scene中只有一个几何体，直接使用
                if len(loaded_object.geometry) == 1:
                    mesh = list(loaded_object.geometry.values())[0]
                else:
                    # 如果有多个几何体，尝试合并为单一网格
                    meshes_to_combine = [
                        geom
                        for geom in loaded_object.geometry.values()
                        if isinstance(geom, trimesh.Trimesh)
                    ]
                    if meshes_to_combine:
                        mesh = trimesh.util.concatenate(meshes_to_combine)
                    else:
                        if verbose:
                            print("  Scene中没有可用的网格几何体")
                        return None
            else:
                if verbose:
                    print("  Scene对象为空")
                return None
        elif isinstance(loaded_object, trimesh.Trimesh):
            # 如果返回Trimesh对象，直接使用
            mesh = loaded_object
        else:
            if verbose:
                print(f"  不支持的网格类型: {type(loaded_object)}")
            return None

        # 检查是否需要应用坐标变换
        if hasattr(collision, "origin") and collision.origin is not None:
            # 应用4x4变换矩阵到网格的所有顶点
            mesh.apply_transform(collision.origin)

        return mesh

    except Exception as e:
        if verbose:
            print(f"  Failed to load mesh {mesh_path}: {e}")
        return None


def _collect_link_meshes(link, verbose=True):
    """
    收集连杆的所有碰撞网格并合并

    Args:
        link: 连杆对象
        verbose (bool): 是否输出详细信息

    Returns:
        trimesh.Trimesh or None: 合并后的网格，失败时返回None
    """
    # 检查当前连杆是否包含碰撞几何信息
    if not link.collisions:
        return None

    meshes = []

    # 遍历当前连杆的所有碰撞几何体
    for collision in link.collisions:
        mesh = _load_mesh_from_collision(collision, verbose)
        if mesh is not None:
            meshes.append(mesh)

    # 如果该连杆没有有效的网格数据，返回None
    if not meshes:
        return None

    # 处理多网格合并：一个连杆可能由多个网格组成
    if len(meshes) > 1:
        # 使用trimesh的concatenate函数合并多个网格为单一网格
        combined_mesh = trimesh.util.concatenate(meshes)
    else:
        # 如果只有一个网格，直接使用
        combined_mesh = meshes[0]

    # 检查合并后的网格是否为空
    if combined_mesh.is_empty:
        return None

    if verbose:
        print(
            f"  Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces"
        )

    return combined_mesh


def _perform_convex_decomposition(mesh, coacd_params, verbose=True):
    """
    对网格进行凸分解

    Args:
        mesh: trimesh对象
        coacd_params (dict): CoACD参数配置
        verbose (bool): 是否输出详细信息

    Returns:
        list: 凸包列表，每个元素为(vertices, faces)
    """
    # 分析网格复杂度
    mesh_analysis = analyze_mesh_complexity(mesh, verbose=verbose)

    # 只对凸性比率 <= 0.7 的网格执行凸分解
    convexity_ratio = mesh_analysis.get("convexity_ratio", 1.0)

    if convexity_ratio <= 0.2:
        # 执行凸分解
        if verbose:
            print(f"  执行凸分解 (凸性比率: {convexity_ratio:.3f})")

        # 准备CoACD输入数据
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)

        # 执行凸分解
        try:
            # 创建CoACD Mesh对象
            coacd_mesh = coacd.Mesh(vertices, faces)
            convex_hulls = coacd.run_coacd(coacd_mesh, **coacd_params)
            if verbose:
                print(f"  CoACD generated {len(convex_hulls)} convex hulls")
        except Exception as e:
            if verbose:
                print(f"  CoACD failed: {e}, falling back to single convex hull")
            # 如果CoACD失败，使用原始网格的凸包
            convex_hull = mesh.convex_hull
            convex_hulls = [(convex_hull.vertices, convex_hull.faces)]
    else:
        # 凸性比率 > 0.7，直接使用原始网格的凸包
        if verbose:
            print(f"  跳过凸分解，直接使用凸包 (凸性比率: {convexity_ratio:.3f})")
        convex_hull = mesh.convex_hull
        convex_hulls = [(convex_hull.vertices, convex_hull.faces)]

    return convex_hulls


def _compute_best_obb_from_convex_hulls(convex_hulls, verbose=True):
    """
    从多个凸包中计算最优的OBB

    Args:
        convex_hulls: 凸包列表，每个元素为(vertices, faces)
        verbose (bool): 是否输出详细信息

    Returns:
        tuple: (best_obb, min_volume) 或 (None, float('inf'))
    """
    best_obb = None
    min_volume = float("inf")

    for i, (conv_vertices, conv_faces) in enumerate(convex_hulls):
        try:
            # 将凸包转换为Open3D网格
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(conv_vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(conv_faces)

            # 确保网格有效性
            if len(o3d_mesh.vertices) < 4:  # 至少需要4个顶点形成四面体
                continue

            # 使用Open3D计算最小OBB
            obb_o3d = o3d_mesh.get_minimal_oriented_bounding_box()

            # 计算OBB体积
            extent = obb_o3d.extent
            volume = extent[0] * extent[1] * extent[2]

            if verbose:
                print(f"    Convex hull {i}: volume={volume:.6f}, extent={extent}")

            # 选择体积最小的OBB
            if volume < min_volume and volume > 1e-12:  # 避免退化情况
                min_volume = volume
                best_obb = obb_o3d

        except Exception as e:
            if verbose:
                print(f"    Failed to compute OBB for convex hull {i}: {e}")
            continue

    return best_obb, min_volume


def _create_obb_dict(link_name, obb_o3d, volume):
    """
    创建OBB结果字典

    Args:
        link_name (str): 连杆名称
        obb_o3d: Open3D OBB对象
        volume (float): OBB体积

    Returns:
        dict: OBB信息字典
    """
    # 提取OBB参数
    center = np.array(obb_o3d.center)
    rotation_matrix = np.array(obb_o3d.R)
    extent = np.array(obb_o3d.extent)

    # 构建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = center

    return {
        "link_name": link_name,
        "position": center,
        "rotation_matrix": rotation_matrix,
        "extents": extent,
        "transform": transform,
        "volume": volume,
    }


def _create_zero_obb(link_name):
    """
    为虚拟连杆创建零大小的OBB

    Args:
        link_name (str): 连杆名称

    Returns:
        dict: 零大小的OBB信息字典
    """
    zero_center = np.array([0.0, 0.0, 0.0])
    identity_rotation = np.eye(3)
    zero_extents = np.array([0.0, 0.0, 0.0])

    # 构建4x4变换矩阵
    zero_transform = np.eye(4)
    zero_transform[:3, :3] = identity_rotation
    zero_transform[:3, 3] = zero_center

    return {
        "link_name": link_name,
        "position": zero_center,
        "rotation_matrix": identity_rotation,
        "extents": zero_extents,
        "transform": zero_transform,
        "volume": 0.0,
    }


def calculate_link_obbs(robot_urdf_path, coacd_params=None, verbose=True):
    """
    Calculate Oriented Bounding Boxes (OBB) for each link in the robot using CoACD and Open3D.

    This function:
    1. Loads robot URDF and extracts collision meshes for each link
    2. Uses CoACD to decompose each mesh into convex components
    3. Uses Open3D to calculate minimal OBB for each convex component
    4. Returns the best (smallest volume) OBB for each link

    Args:
        robot_urdf_path (str): Path to the robot URDF file
        coacd_params (dict, optional): CoACD参数配置，如果None则使用默认值
        verbose (bool): 是否输出详细日志信息

    Returns:
        List[dict]: 包含每个连杆OBB信息的字典列表，每个字典包含:
            - link_name (str): 连杆名称
            - position (np.ndarray): OBB中心位置 [x, y, z]
            - rotation_matrix (np.ndarray): 3x3旋转矩阵
            - extents (np.ndarray): OBB尺寸 [长, 宽, 高]
            - transform (np.ndarray): 4x4变换矩阵
            - volume (float): OBB体积
    """
    # 检查依赖库
    if not HAS_ALL_LIBS:
        if verbose:
            print(f"Warning: Missing required libraries: {', '.join(MISSING_LIBS)}")
            print("Please install: pip install " + " ".join(MISSING_LIBS))
        return []

    # 使用默认参数如果没有提供
    if coacd_params is None:
        coacd_params = get_default_coacd_params()

    try:
        # 加载URDF文件
        robot, temp_urdf_path = _load_robot_from_urdf(robot_urdf_path, verbose)

        try:
            # 初始化OBB结果列表
            obbs = []

            # 遍历机器人模型中的所有连杆
            for link_name, link in robot.link_map.items():
                # 处理虚拟连杆（无碰撞几何）
                if not link.collisions:
                    if verbose:
                        print(
                            f"Processing virtual link '{link_name}' (no collision geometry)..."
                        )

                    # obbs.append(_create_zero_obb(link_name))
                    # if verbose:
                    #     print(f"  Created zero-size OBB for virtual link {link_name}")
                    # continue

                if verbose:
                    print(f"Processing link '{link_name}'...")

                # 收集并合并连杆的所有网格
                combined_mesh = _collect_link_meshes(link, verbose)
                if combined_mesh is None:
                    continue

                try:
                    # 进行凸分解
                    convex_hulls = _perform_convex_decomposition(
                        combined_mesh, coacd_params, verbose
                    )

                    # 计算最优OBB
                    best_obb, min_volume = _compute_best_obb_from_convex_hulls(
                        convex_hulls, verbose
                    )

                    if best_obb is not None:
                        if verbose:
                            print(f"  Final OBB volume: {min_volume:.6f}")

                        # 创建OBB结果字典并添加到列表
                        obb_dict = _create_obb_dict(link_name, best_obb, min_volume)
                        obbs.append(obb_dict)

                        if verbose:
                            print(f"  Successfully computed OBB for {link_name}")
                    else:
                        if verbose:
                            print(f"  Failed to compute any valid OBB for {link_name}")

                except Exception as e:
                    # 如果OBB计算过程中出现任何异常，跳过该连杆
                    if verbose:
                        print(f"  OBB calculation failed for {link_name}: {e}")
                    continue

            # 返回所有成功计算的OBB信息列表
            return obbs

        finally:
            # 清理临时文件
            if os.path.exists(temp_urdf_path):
                os.unlink(temp_urdf_path)

    except Exception as e:
        if verbose:
            print(f"Error in calculate_link_obbs: {e}")
        return []


def calculate_single_mesh_obb(mesh, coacd_params=None, verbose=False):
    """
    为单个网格计算最优的有向包围盒(OBB)

    使用CoACD进行凸分解，然后为每个凸包计算OBB，最终返回体积最小的OBB。

    Args:
        mesh (trimesh.Trimesh): 输入的三维网格
        coacd_params (dict, optional): CoACD参数配置，如果None则使用默认值
        verbose (bool): 是否输出详细信息

    Returns:
        dict or None: OBB信息字典，包含position, rotation_matrix, extents, transform, volume
                     如果计算失败则返回None
    """
    # 检查依赖库
    if not HAS_ALL_LIBS:
        if verbose:
            print(f"Warning: Missing required libraries: {', '.join(MISSING_LIBS)}")
        return None

    # 使用默认参数
    if coacd_params is None:
        coacd_params = get_default_coacd_params()

    try:
        # 检查网格有效性
        if mesh.is_empty:
            if verbose:
                print("Input mesh is empty")
            return None

        # 执行凸分解
        convex_hulls = _perform_convex_decomposition(mesh, coacd_params, verbose)

        # 计算最优OBB
        best_obb, min_volume = _compute_best_obb_from_convex_hulls(
            convex_hulls, verbose
        )

        # 返回结果
        if best_obb is not None:
            return _create_obb_dict("single_mesh", best_obb, min_volume)
        else:
            if verbose:
                print("Failed to compute any valid OBB")
            return None

    except Exception as e:
        if verbose:
            print(f"Single mesh OBB calculation failed: {e}")
        return None


if __name__ == "__main__":
    # 简单的测试代码
    print("OBB Calculator Module")
    print(f"Dependencies available: {HAS_ALL_LIBS}")
    if not HAS_ALL_LIBS:
        print(f"Missing libraries: {', '.join(MISSING_LIBS)}")
        print("Install with: pip install " + " ".join(MISSING_LIBS))
