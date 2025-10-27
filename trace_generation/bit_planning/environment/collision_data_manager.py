import pickle


class CollisionDataManager:
    """碰撞数据管理类，负责碰撞数据的存储、计算和统计"""

    def __init__(self, enable_sphere=False):
        """
        初始化碰撞数据管理器

        Args:
            enable_sphere: 是否启用球体碰撞检测
        """
        # 碰撞数据收集
        self.obb_link_data = []
        self.obb_link_coll_data = []
        self.sphere_link_data = []
        self.sphere_link_coll_data = []

        # 边检查统计
        self.edge_fp_call_count = 0
        self.edge_fp_rejected_by_endpoints = 0

        self.enable_sphere = enable_sphere

    def reset(self):
        """重置所有碰撞数据和统计量"""
        self.obb_link_data.clear()
        self.obb_link_coll_data.clear()
        self.sphere_link_data.clear()
        self.sphere_link_coll_data.clear()
        self.edge_fp_call_count = 0
        self.edge_fp_rejected_by_endpoints = 0

    def _store_collision_data(
        self, link_coords, link_colls, sphere_coords, sphere_colls, is_edge=True
    ):
        """
        存储碰撞数据到相应的数据结构中

        Args:
            link_coords: link坐标数据
            link_colls: link碰撞标签
            sphere_coords: sphere坐标数据
            sphere_colls: sphere碰撞标签
            is_edge: 是否为边数据（True）还是单点数据（False）
        """
        if not link_coords:
            return

        if is_edge:
            # 边数据：添加到现有边的数据结构中
            self.obb_link_data.append(link_coords)
            self.obb_link_coll_data.append(link_colls)
            self.sphere_link_data.append(sphere_coords)
            self.sphere_link_coll_data.append(sphere_colls)
        else:
            # 单点数据：包装成单元素列表
            self.obb_link_data.append([link_coords])
            self.obb_link_coll_data.append([link_colls])
            self.sphere_link_data.append([sphere_coords])
            self.sphere_link_coll_data.append([sphere_colls])

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
            tuple: (obb_link_ratio, obb_pose_ratio, obb_edge_ratio, sphere_link_ratio, sphere_pose_ratio, sphere_edge_ratio)
        """
        # 计算OBB碰撞率
        obb_link_ratio, obb_pose_ratio, obb_edge_ratio = (
            self._calculate_collision_ratios(self.obb_link_coll_data)
        )

        # 计算Sphere碰撞率
        sphere_link_ratio, sphere_pose_ratio, sphere_edge_ratio = (
            self._calculate_collision_ratios(self.sphere_link_coll_data)
        )

        return (
            obb_link_ratio,
            obb_pose_ratio,
            obb_edge_ratio,
            sphere_link_ratio,
            sphere_pose_ratio,
            sphere_edge_ratio,
        )

    def count_edge_level_discrepancies(self):
        """
        在 pose 层面比较 OBB 与 Sphere 的碰撞结果，统计不一致的pose数。

        规则：
        - 一个 pose 被视为碰撞（collided）当且仅当该 pose 的任一 link 存在碰撞（coll_value == 0）。
        - 同一索引下的 obb 和 sphere pose 比较：
            * both_collided: 两者都判断为碰撞
            * both_free: 两者都判断为无碰撞
            * obb_only: 仅 OBB 判断为碰撞
            * sphere_only: 仅 Sphere 判断为碰撞
        - 如果两者 pose 数量不一致，超出的部分也会被计入对应的 only 类别（视为另一方缺失 -> 视为无碰撞）

        Returns:
            dict: 包含统计字段的字典，字段包括：
                total_compared: 实际比较的pose数量
                both_collided, both_free, obb_only, sphere_only, mismatches_total
        """
        obb_edges = self.obb_link_coll_data or []
        sphere_edges = self.sphere_link_coll_data or []

        len_obb = len(obb_edges)
        len_sphere = len(sphere_edges)
        total_edges = max(len_obb, len_sphere)

        both_collided = 0
        both_free = 0
        obb_only = 0
        sphere_only = 0
        total_poses_compared = 0

        for i in range(total_edges):
            # 获取 edge 的 pose 列表，若超出则视为空（无碰撞）
            obb_edge = obb_edges[i] if i < len_obb else []
            sphere_edge = sphere_edges[i] if i < len_sphere else []

            # 获取该edge中的pose数量
            len_poses_obb = len(obb_edge)
            len_poses_sphere = len(sphere_edge)
            max_poses = max(len_poses_obb, len_poses_sphere)

            for j in range(max_poses):
                total_poses_compared += 1

                # 获取 pose 的 link 碰撞列表，若超出则视为空（无碰撞）
                obb_pose = obb_edge[j] if j < len_poses_obb else []
                sphere_pose = sphere_edge[j] if j < len_poses_sphere else []

                # 判断 pose 是否碰撞（任一 link 中 coll_value == 0 即视为碰撞）
                obb_pose_collided = (
                    any(coll_value == 0 for coll_value in obb_pose)
                    if obb_pose
                    else False
                )

                sphere_pose_collided = (
                    any(coll_value == 0 for coll_value in sphere_pose)
                    if sphere_pose
                    else False
                )

                if obb_pose_collided and sphere_pose_collided:
                    both_collided += 1
                elif not obb_pose_collided and not sphere_pose_collided:
                    both_free += 1
                elif obb_pose_collided and not sphere_pose_collided:
                    obb_only += 1
                elif sphere_pose_collided and not obb_pose_collided:
                    sphere_only += 1

        mismatches = obb_only + sphere_only

        print(
            f"Pose Level Discrepancies: {mismatches} mismatches over {total_poses_compared} poses. obb only {obb_only}, sphere only {sphere_only}!!!"
        )
        return {
            "total_compared": total_poses_compared,
            "both_collided": both_collided,
            "both_free": both_free,
            "obb_only": obb_only,
            "sphere_only": sphere_only,
            "mismatches_total": mismatches,
        }

    def save_collision_data(self, link_output_file, sphere_output_file):
        """保存碰撞数据到文件"""
        self.count_edge_level_discrepancies()

        # 输出边检查统计信息
        actual_edge_checks = (
            self.edge_fp_call_count - self.edge_fp_rejected_by_endpoints
        )
        rejection_rate = (
            (self.edge_fp_rejected_by_endpoints / self.edge_fp_call_count * 100)
            if self.edge_fp_call_count > 0
            else 0.0
        )

        print(
            f"边检查统计: 总调用次数 {self.edge_fp_call_count}, 因端点无效排除 {self.edge_fp_rejected_by_endpoints} ({rejection_rate:.1f}%), 实际边检查 {actual_edge_checks}"
        )

        with open(link_output_file, "wb") as f:
            pickle.dump((self.obb_link_data, self.obb_link_coll_data), f)

        with open(sphere_output_file, "wb") as f:
            pickle.dump((self.sphere_link_data, self.sphere_link_coll_data), f)
        (
            obb_ratio,
            obb_pose_ratio,
            obb_edge_ratio,
            sphere_ratio,
            sphere_pose_ratio,
            sphere_edge_ratio,
        ) = self.get_collision_ratio()

        # 统计有效边数（包含多个pose的边）
        valid_obb_edges = sum(1 for edge in self.obb_link_coll_data if len(edge) > 1)
        valid_sphere_edges = sum(
            1 for edge in self.sphere_link_coll_data if len(edge) > 1
        )

        print(
            f"✓ 保存Link数据: {valid_obb_edges} 条有效边 (总{len(self.obb_link_data)}条), OBB Link 碰撞率: {obb_ratio:.4f} , pose 碰撞率: {obb_pose_ratio:.4f} , edge 碰撞率: {obb_edge_ratio:.4f}"
        )
        print(
            f"✓ 保存Sphere数据: {valid_sphere_edges} 条有效边 (总{len(self.sphere_link_data)}条), Sphere碰撞率: {sphere_ratio:.4f} , pose 碰撞率: {sphere_pose_ratio:.4f} , edge 碰撞率: {sphere_edge_ratio:.4f}"
        )

    def get_edge_check_statistics(self):
        """
        获取边检查统计信息

        Returns:
            dict: 包含边检查统计的字典
        """
        actual_edge_checks = (
            self.edge_fp_call_count - self.edge_fp_rejected_by_endpoints
        )
        rejection_rate = (
            (self.edge_fp_rejected_by_endpoints / self.edge_fp_call_count * 100)
            if self.edge_fp_call_count > 0
            else 0.0
        )

        return {
            "total_edge_fp_calls": self.edge_fp_call_count,
            "rejected_by_endpoints": self.edge_fp_rejected_by_endpoints,
            "actual_edge_checks": actual_edge_checks,
            "rejection_rate_percent": rejection_rate,
        }