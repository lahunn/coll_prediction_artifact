# type: ignore
"""
OBB Calculator - 机器人连杆有向包围盒计算模块

该模块使用CoACD和Open3D库为机器人连杆计算精确的有向包围盒(OBB)。

主要功能:
1. 使用CoACD进行凸分解，将复杂网格分解为多个凸包
2. 使用Open3D计算每个凸包的最小有向包围盒
3. 选择体积最小的OBB作为最终结果

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
        urdf_dir = Path(robot_urdf_path).parent

        # 读取并修改URDF内容以替换package://路径
        with open(robot_urdf_path, "r") as f:
            urdf_content = f.read()

        # 替换package://路径为绝对路径
        modified_content = urdf_content.replace("package://", str(urdf_dir) + "/")

        # 写入临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False
        ) as tmp_file:
            tmp_file.write(modified_content)
            temp_urdf_path = tmp_file.name

        try:
            # 使用yourdfpy库加载并解析URDF文件
            robot = yourdfpy.URDF.load(temp_urdf_path)

            # 初始化OBB结果列表
            obbs = []

            # 遍历机器人模型中的所有连杆
            for link_name, link in robot.link_map.items():
                # 检查当前连杆是否包含碰撞几何信息
                if not link.collisions:
                    # 为虚拟连杆（无碰撞几何）创建零大小的OBB
                    if verbose:
                        print(
                            f"Processing virtual link '{link_name}' (no collision geometry)..."
                        )

                    # 创建零大小的OBB
                    zero_center = np.array([0.0, 0.0, 0.0])
                    identity_rotation = np.eye(3)
                    zero_extents = np.array([0.0, 0.0, 0.0])

                    # 构建4x4变换矩阵
                    zero_transform = np.eye(4)
                    zero_transform[:3, :3] = identity_rotation
                    zero_transform[:3, 3] = zero_center

                    # 添加零大小的OBB到结果列表
                    obbs.append(
                        {
                            "link_name": link_name,  # 连杆名称
                            "position": zero_center,  # OBB中心位置
                            "rotation_matrix": identity_rotation,  # 3x3旋转矩阵
                            "extents": zero_extents,  # OBB尺寸
                            "transform": zero_transform,  # 完整的4x4变换矩阵
                            "volume": 0.0,  # OBB体积
                        }
                    )

                    if verbose:
                        print(f"  Created zero-size OBB for virtual link {link_name}")
                    continue

                if verbose:
                    print(f"Processing link '{link_name}'...")

                # 初始化网格列表，用于收集该连杆的所有碰撞网格
                meshes = []

                # 遍历当前连杆的所有碰撞几何体
                for collision in link.collisions:
                    # 检查碰撞几何是否为网格类型
                    if hasattr(collision.geometry, "mesh") and collision.geometry.mesh:
                        # 获取网格文件的路径
                        mesh_path = collision.geometry.mesh.filename

                        # 验证网格文件是否存在
                        if Path(mesh_path).exists():
                            try:
                                # 使用trimesh库加载3D网格文件
                                loaded_object = trimesh.load(str(mesh_path))

                                # 处理不同的返回类型
                                if isinstance(loaded_object, trimesh.Scene):
                                    # 如果返回Scene对象，需要提取其中的网格
                                    if loaded_object.geometry:
                                        # 如果Scene中只有一个几何体，直接使用
                                        if len(loaded_object.geometry) == 1:
                                            mesh = list(
                                                loaded_object.geometry.values()
                                            )[0]
                                        else:
                                            # 如果有多个几何体，尝试合并为单一网格
                                            meshes_to_combine = [
                                                geom
                                                for geom in loaded_object.geometry.values()
                                                if isinstance(geom, trimesh.Trimesh)
                                            ]
                                            if meshes_to_combine:
                                                mesh = trimesh.util.concatenate(
                                                    meshes_to_combine
                                                )
                                            else:
                                                if verbose:
                                                    print(
                                                        "  Scene中没有可用的网格几何体"
                                                    )
                                                continue
                                    else:
                                        if verbose:
                                            print("  Scene对象为空")
                                        continue
                                elif isinstance(loaded_object, trimesh.Trimesh):
                                    # 如果返回Trimesh对象，直接使用
                                    mesh = loaded_object
                                else:
                                    if verbose:
                                        print(
                                            f"  不支持的网格类型: {type(loaded_object)}"
                                        )
                                    continue

                                # 检查是否需要应用坐标变换
                                if (
                                    hasattr(collision, "origin")
                                    and collision.origin is not None
                                ):
                                    # 应用4x4变换矩阵到网格的所有顶点
                                    mesh.apply_transform(collision.origin)

                                # 将处理好的网格添加到列表中
                                meshes.append(mesh)
                            except Exception as e:
                                if verbose:
                                    print(f"  Failed to load mesh {mesh_path}: {e}")
                                continue

                # 如果该连杆没有有效的网格数据，跳过OBB计算
                if not meshes:
                    continue

                try:
                    # 处理多网格合并：一个连杆可能由多个网格组成
                    if len(meshes) > 1:
                        # 使用trimesh的concatenate函数合并多个网格为单一网格
                        combined_mesh = trimesh.util.concatenate(meshes)
                    else:
                        # 如果只有一个网格，直接使用
                        combined_mesh = meshes[0]

                    # 检查合并后的网格是否为空
                    if combined_mesh.is_empty:
                        continue

                    if verbose:
                        print(
                            f"  Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces"
                        )

                    # === 分析网格复杂度 ===
                    mesh_analysis = analyze_mesh_complexity(
                        combined_mesh, verbose=verbose
                    )

                    # 只对凸性比率 <= 0.95 的网格执行凸分解
                    convexity_ratio = mesh_analysis.get("convexity_ratio", 1.0)
                    # if convexity_ratio <= 0.95:
                    if convexity_ratio <= 0.7:
                        # === Step 1: 使用CoACD进行凸分解 ===
                        if verbose:
                            print(f"  执行凸分解 (凸性比率: {convexity_ratio:.3f})")

                        # 准备CoACD输入数据
                        vertices = np.array(combined_mesh.vertices, dtype=np.float64)
                        faces = np.array(combined_mesh.faces, dtype=np.int32)

                        # 执行凸分解
                        try:
                            # 创建CoACD Mesh对象
                            coacd_mesh = coacd.Mesh(vertices, faces)
                            convex_hulls = coacd.run_coacd(coacd_mesh, **coacd_params)
                            if verbose:
                                print(
                                    f"  CoACD generated {len(convex_hulls)} convex hulls"
                                )
                        except Exception as e:
                            if verbose:
                                print(
                                    f"  CoACD failed: {e}, falling back to single convex hull"
                                )
                            # 如果CoACD失败，使用原始网格的凸包
                            convex_hull = combined_mesh.convex_hull
                            convex_hulls = [(convex_hull.vertices, convex_hull.faces)]
                    else:
                        # 凸性比率 > 0.95，直接使用原始网格的凸包
                        if verbose:
                            print(
                                f"  跳过凸分解，直接使用凸包 (凸性比率: {convexity_ratio:.3f})"
                            )
                        convex_hull = combined_mesh.convex_hull
                        convex_hulls = [(convex_hull.vertices, convex_hull.faces)]

                    # === Step 2: 为每个凸包计算OBB并选择最优的 ===
                    best_obb = None
                    min_volume = float("inf")

                    for i, (conv_vertices, conv_faces) in enumerate(convex_hulls):
                        try:
                            # 将凸包转换为Open3D网格
                            o3d_mesh = o3d.geometry.TriangleMesh()
                            o3d_mesh.vertices = o3d.utility.Vector3dVector(
                                conv_vertices
                            )
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
                                print(
                                    f"    Convex hull {i}: volume={volume:.6f}, extent={extent}"
                                )

                            # 选择体积最小的OBB
                            if volume < min_volume and volume > 1e-12:  # 避免退化情况
                                min_volume = volume
                                best_obb = obb_o3d

                        except Exception as e:
                            if verbose:
                                print(
                                    f"    Failed to compute OBB for convex hull {i}: {e}"
                                )
                            continue

                    # 如果没有找到有效的OBB，尝试使用整个合并网格
                    if best_obb is None:
                        if verbose:
                            print(
                                "  No valid convex hull OBB found, using combined mesh"
                            )
                        try:
                            # 转换为Open3D网格
                            o3d_mesh = o3d.geometry.TriangleMesh()
                            o3d_mesh.vertices = o3d.utility.Vector3dVector(
                                combined_mesh.vertices
                            )
                            o3d_mesh.triangles = o3d.utility.Vector3iVector(
                                combined_mesh.faces
                            )

                            # 计算OBB
                            best_obb = o3d_mesh.get_minimal_oriented_bounding_box()
                            min_volume = np.prod(best_obb.extent)
                        except Exception as e:
                            if verbose:
                                print(f"  Failed to compute OBB for combined mesh: {e}")
                            continue

                    if best_obb is not None:
                        if verbose:
                            print(f"  Final OBB volume: {min_volume:.6f}")

                        # 提取OBB参数
                        center = np.array(best_obb.center)
                        rotation_matrix = np.array(best_obb.R)
                        extent = np.array(best_obb.extent)

                        # 构建4x4变换矩阵
                        transform = np.eye(4)
                        transform[:3, :3] = rotation_matrix
                        transform[:3, 3] = center

                        # 将计算结果存储为字典并添加到结果列表
                        obbs.append(
                            {
                                "link_name": link_name,  # 连杆名称
                                "position": center,  # OBB中心位置
                                "rotation_matrix": rotation_matrix,  # 3x3旋转矩阵
                                "extents": extent,  # OBB尺寸
                                "transform": transform,  # 完整的4x4变换矩阵
                                "volume": min_volume,  # OBB体积
                            }
                        )

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
    为单个网格计算OBB

    Args:
        mesh: trimesh对象
        coacd_params (dict, optional): CoACD参数配置
        verbose (bool): 是否输出详细信息

    Returns:
        dict or None: OBB信息字典或None（如果计算失败）
    """
    if not HAS_ALL_LIBS:
        return None

    if coacd_params is None:
        coacd_params = get_default_coacd_params()

    try:
        if mesh.is_empty:
            return None

        # CoACD凸分解
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)

        try:
            # 创建CoACD Mesh对象
            coacd_mesh = coacd.Mesh(vertices, faces)
            convex_hulls = coacd.run_coacd(coacd_mesh, **coacd_params)
        except Exception as e:
            if verbose:
                print(f"CoACD failed: {e}, using single convex hull")
            convex_hull = mesh.convex_hull
            convex_hulls = [(convex_hull.vertices, convex_hull.faces)]

        # 找到最小体积的OBB
        best_obb = None
        min_volume = float("inf")

        for conv_vertices, conv_faces in convex_hulls:
            try:
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(conv_vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(conv_faces)

                if len(o3d_mesh.vertices) < 4:
                    continue

                obb_o3d = o3d_mesh.get_minimal_oriented_bounding_box()
                volume = np.prod(obb_o3d.extent)

                if volume < min_volume and volume > 1e-12:
                    min_volume = volume
                    best_obb = obb_o3d

            except Exception:
                continue

        if best_obb is not None:
            center = np.array(best_obb.center)
            rotation_matrix = np.array(best_obb.R)
            extent = np.array(best_obb.extent)

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = center

            return {
                "position": center,
                "rotation_matrix": rotation_matrix,
                "extents": extent,
                "transform": transform,
                "volume": min_volume,
            }

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
