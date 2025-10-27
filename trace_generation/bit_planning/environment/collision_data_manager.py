import pickle


class CollisionDataManager:
    """碰撞数据管理类，负责碰撞数据的存储、计算和统计"""

    def __init__(self):
        """
        初始化碰撞数据管理器
        """
        # 碰撞数据收集
        self.obb_link_data = []
        self.obb_link_coll_data = []

        # 碰撞统计
        self.collision_check_count = 0
        self.collision_time = 0.0

    def reset(self):
        """重置所有碰撞数据和统计量"""
        self.obb_link_data.clear()
        self.obb_link_coll_data.clear()
        self.edge_fp_call_count = 0
        self.collision_check_count = 0
        self.collision_time = 0

    def _store_collision_data(self, link_coords, link_colls, is_edge=True):
        """
        存储碰撞数据到相应的数据结构中

        Args:
            link_coords: link坐标数据
            link_colls: link碰撞标签
            is_edge: 是否为边数据（True）还是单点数据（False）
        """
        if not link_coords:
            return

        if is_edge:
            # 边数据：添加到现有边的数据结构中
            self.obb_link_data.append(link_coords)
            self.obb_link_coll_data.append(link_colls)
        else:
            # 单点数据：包装成单元素列表
            self.obb_link_data.append([link_coords])
            self.obb_link_coll_data.append([link_colls])

    def _calculate_collision_ratios(self, link_coll_data):
        """
        计算给定碰撞数据的各层级碰撞率

        Args:
            link_coll_data: 碰撞数据列表

        Returns:
            tuple: (link_ratio, pose_ratio, edge_ratio)
        """
        link_ratio = 0.0
        pose_ratio = 0.0
        edge_ratio = 0.0

        if link_coll_data:
            total_links = 0
            collided_links = 0
            total_poses = 0
            collided_poses = 0
            total_edges = len(link_coll_data)
            collided_edges = 0

            for edge_colls in link_coll_data:
                is_edge_collided = False
                for pose_colls in edge_colls:
                    total_poses += 1
                    is_pose_collided = False
                    for coll_value in pose_colls:
                        total_links += 1
                        if coll_value == 0:
                            collided_links += 1
                            is_pose_collided = (
                                True  # 只要有一个link碰撞, pose就视为碰撞
                            )

                    if is_pose_collided:
                        collided_poses += 1
                        is_edge_collided = True  # 只要有一个pose碰撞, edge就视为碰撞

                if is_edge_collided:
                    collided_edges += 1

            link_ratio = collided_links / total_links if total_links > 0 else 0.0
            pose_ratio = collided_poses / total_poses if total_poses > 0 else 0.0
            edge_ratio = collided_edges / total_edges if total_edges > 0 else 0.0

        return link_ratio, pose_ratio, edge_ratio

    def get_collision_ratio(self):
        """计算各层级的碰撞率

        Returns:
            tuple: (obb_link_ratio, obb_pose_ratio, obb_edge_ratio)
        """
        # 计算OBB碰撞率
        obb_link_ratio, obb_pose_ratio, obb_edge_ratio = (
            self._calculate_collision_ratios(self.obb_link_coll_data)
        )

        return (
            obb_link_ratio,
            obb_pose_ratio,
            obb_edge_ratio,
        )

    def save_collision_data(self, link_output_file):
        """保存碰撞数据到文件"""
        print(f"边检查统计: 总调用次数 {self.edge_fp_call_count}")

        with open(link_output_file, "wb") as f:
            pickle.dump((self.obb_link_data, self.obb_link_coll_data), f)

        (
            obb_ratio,
            obb_pose_ratio,
            obb_edge_ratio,
        ) = self.get_collision_ratio()

        # 统计有效边数（包含多个pose的边）
        valid_obb_edges = sum(1 for edge in self.obb_link_coll_data if len(edge) > 1)

        print(
            f"✓ 保存Link数据: {valid_obb_edges} 条有效边 (总{len(self.obb_link_data)}条), OBB Link 碰撞率: {obb_ratio:.4f} , pose 碰撞率: {obb_pose_ratio:.4f} , edge 碰撞率: {obb_edge_ratio:.4f}"
        )

    def get_edge_check_statistics(self):
        """
        获取边检查统计信息

        Returns:
            dict: 包含边检查统计的字典
        """

        return {
            "total_edge_fp_calls": self.edge_fp_call_count,
        }
