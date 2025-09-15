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
except ImportError as e:
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
        print(f"  网格分析结果:")
        print(f"    顶点数: {analysis['vertex_count']}, 面数: {analysis['face_count']}")
        print(
            f"    水密性: {analysis['is_watertight']}, 绕向一致: {analysis['is_winding_consistent']}"
        )
        print(f"    凸性比率: {analysis.get('convexity_ratio', 'N/A'):.4f}")
        print(f"    包围盒体积比: {analysis.get('bbox_to_volume_ratio', 'N/A'):.4f}")
        if analysis.get("convexity_ratio", 1.0) > 0.95:
            print(f"    ⚠️  网格已经非常接近凸形状，跳过凸分解")

    return analysis


def _create_filename_handler(urdf_dir):
    """创建自定义文件名处理器"""

    def handler(filename):
        if filename.startswith("package://"):
            return str(urdf_dir / filename.replace("package://", ""))
        elif not filename.startswith("/"):
            return str(urdf_dir / filename)
        else:
            return filename

    return handler


def _create_zero_obb(link_name):
    """为虚拟连杆创建零大小的OBB"""
    zero_center = np.array([0.0, 0.0, 0.0])
    identity_rotation = np.eye(3)
    zero_extents = np.array([0.0, 0.0, 0.0])
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


def _load_mesh_from_collision(collision, verbose=False):
    """从碰撞几何体加载网格"""
    if not (hasattr(collision.geometry, "mesh") and collision.geometry.mesh):
        return None

    mesh_path = collision.geometry.mesh.filename
    if not Path(mesh_path).exists():
        return None

    try:
        loaded_object = trimesh.load(str(mesh_path))

        # 处理不同的返回类型
        if isinstance(loaded_object, trimesh.Scene):
            if not loaded_object.geometry:
                return None
            if len(loaded_object.geometry) == 1:
                mesh = list(loaded_object.geometry.values())[0]
            else:
                meshes_to_combine = [
                    geom
                    for geom in loaded_object.geometry.values()
                    if isinstance(geom, trimesh.Trimesh)
                ]
                if meshes_to_combine:
                    mesh = trimesh.util.concatenate(meshes_to_combine)
                else:
                    return None
        elif isinstance(loaded_object, trimesh.Trimesh):
            mesh = loaded_object
        else:
            return None

        # 应用变换
        if hasattr(collision, "origin") and collision.origin is not None:
            mesh.apply_transform(collision.origin)

        return mesh
    except Exception as e:
        if verbose:
            print(f"  Failed to load mesh {mesh_path}: {e}")
        return None


def _perform_convex_decomposition(mesh, coacd_params, verbose=False):
    """执行凸分解"""
    mesh_analysis = analyze_mesh_complexity(mesh, verbose=verbose)
    convexity_ratio = mesh_analysis.get("convexity_ratio", 1.0)

    if convexity_ratio <= 0.7:
        if verbose:
            print(f"  执行凸分解 (凸性比率: {convexity_ratio:.3f})")

        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)

        try:
            coacd_mesh = coacd.Mesh(vertices, faces)
            convex_hulls = coacd.run_coacd(coacd_mesh, **coacd_params)
            if verbose:
                print(f"  CoACD generated {len(convex_hulls)} convex hulls")
            return convex_hulls
        except Exception as e:
            if verbose:
                print(f"  CoACD failed: {e}, falling back to single convex hull")
    else:
        if verbose:
            print(f"  跳过凸分解，直接使用凸包 (凸性比率: {convexity_ratio:.3f})")

    # 回退到单个凸包
    convex_hull = mesh.convex_hull
    return [(convex_hull.vertices, convex_hull.faces)]


def _compute_best_obb_from_hulls(convex_hulls, verbose=False):
    """从凸包列表中计算最佳OBB"""
    best_obb = None
    min_volume = float("inf")

    for i, (conv_vertices, conv_faces) in enumerate(convex_hulls):
        try:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(conv_vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(conv_faces)

            if len(o3d_mesh.vertices) < 4:
                continue

            obb_o3d = o3d_mesh.get_minimal_oriented_bounding_box()
            volume = np.prod(obb_o3d.extent)

            if verbose:
                print(f"    Convex hull {i}: volume={volume:.6f}")

            if volume < min_volume and volume > 1e-12:
                min_volume = volume
                best_obb = obb_o3d
        except Exception as e:
            if verbose:
                print(f"    Failed to compute OBB for convex hull {i}: {e}")

    return best_obb, min_volume


def _compute_obb_from_mesh(mesh, verbose=False):
    """直接从网格计算OBB（备用方案）"""
    try:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        obb_o3d = o3d_mesh.get_minimal_oriented_bounding_box()
        volume = np.prod(obb_o3d.extent)
        return obb_o3d, volume
    except Exception as e:
        if verbose:
            print(f"  Failed to compute OBB from mesh: {e}")
        return None, 0


def _create_obb_dict(link_name, obb_o3d, volume):
    """创建OBB结果字典"""
    center = np.array(obb_o3d.center)
    rotation_matrix = np.array(obb_o3d.R)
    extent = np.array(obb_o3d.extent)

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


def _process_single_link(link_name, link, coacd_params, verbose=False):
    """处理单个连杆的OBB计算"""
    # 处理虚拟连杆
    if not link.collisions:
        if verbose:
            print(f"Processing virtual link '{link_name}' (no collision geometry)...")
        return _create_zero_obb(link_name)

    if verbose:
        print(f"Processing link '{link_name}'...")

    # 加载所有碰撞网格
    meshes = []
    for collision in link.collisions:
        mesh = _load_mesh_from_collision(collision, verbose)
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        return None

    # 合并网格
    combined_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    if combined_mesh.is_empty:
        return None

    if verbose:
        print(
            f"  Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces"
        )

    # 执行凸分解
    convex_hulls = _perform_convex_decomposition(combined_mesh, coacd_params, verbose)

    # 计算最佳OBB
    best_obb, min_volume = _compute_best_obb_from_hulls(convex_hulls, verbose)

    # 如果失败，使用备用方案
    if best_obb is None:
        if verbose:
            print("  No valid convex hull OBB found, using combined mesh")
        best_obb, min_volume = _compute_obb_from_mesh(combined_mesh, verbose)

    if best_obb is not None:
        if verbose:
            print(f"  Final OBB volume: {min_volume:.6f}")
        return _create_obb_dict(link_name, best_obb, min_volume)

    return None


def calculate_link_obbs(robot_urdf_path, coacd_params=None, verbose=True):
    """
    计算机器人所有连杆的OBB

    Args:
        robot_urdf_path (str): URDF文件路径
        coacd_params (dict, optional): CoACD参数配置
        verbose (bool): 是否输出详细信息

    Returns:
        List[dict]: OBB信息列表
    """
    if not HAS_ALL_LIBS:
        if verbose:
            print(f"Warning: Missing required libraries: {', '.join(MISSING_LIBS)}")
        return []

    if coacd_params is None:
        coacd_params = get_default_coacd_params()

    try:
        urdf_dir = Path(robot_urdf_path).parent
        filename_handler = _create_filename_handler(urdf_dir)

        # 加载URDF
        robot = yourdfpy.URDF.load(
            robot_urdf_path,
            filename_handler=filename_handler,
            load_meshes=True,
            load_collision_meshes=True,
            mesh_dir=str(urdf_dir),
        )

        if verbose:
            print(f"  Successfully loaded URDF with {len(robot.link_map)} links")

        # 处理所有连杆
        obbs = []
        for link_name, link in robot.link_map.items():
            try:
                obb_result = _process_single_link(
                    link_name, link, coacd_params, verbose
                )
                if obb_result is not None:
                    obbs.append(obb_result)
                    if verbose:
                        print(f"  Successfully computed OBB for {link_name}")
            except Exception as e:
                if verbose:
                    print(f"  OBB calculation failed for {link_name}: {e}")

        return obbs

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
