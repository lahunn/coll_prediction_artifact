#!/usr/bin/env python3
"""
简单的球体和OBB碰撞检测与CHT训练程序

功能:
1. 在空间中构建一个球体和一个OBB,两者体积相近
2. 加载障碍物环境
3. 随机调整球体和OBB的位置和方向
4. 进行碰撞检测并训练CHT
5. 统计碰撞预测的准确率和召回率

不涉及正向运动学,直接对几何体进行变换
"""

import pybullet as p
import numpy as np
import pickle
from prediction_approaches.collision_prediction_strategies import FixedThresholdStrategy


class SimpleCollisionCHTTrainer:
    """简单的碰撞检测与CHT训练器"""

    def __init__(self, use_gui=False):
        """初始化仿真环境"""
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, 0)
        
        self.obstacle_ids = []
        self.sphere_id = None
        self.obb_id = None
        
        # 定义相近体积的球体和OBB
        # 球体: V = 4/3 * π * r^3, 设r=0.5, V≈0.524
        self.sphere_radius = 0.5
        sphere_volume = (4/3) * np.pi * self.sphere_radius**3
        
        # OBB: V = a * b * c, 设计使体积相近
        # 假设为长方体 0.8 * 0.8 * 0.8 ≈ 0.512
        self.obb_halfextents = [0.2, 0.8, 0.4]
        obb_volume = 8 * np.prod(self.obb_halfextents)
        
        print(f"球体体积: {sphere_volume:.4f}")
        print(f"OBB体积: {obb_volume:.4f}")
        
        # 定义空间采样范围
        self.position_range = [-2.0, 2.0]  # x, y, z范围
        
    def load_obstacles(self, scene_file=None, density=None, num_obstacles=None):
        """加载障碍物环境

        density: 可选, 'low'|'mid'|'high' 映射到默认障碍物数量
        num_obstacles: 可直接传入整数覆盖密度设置
        """
        if scene_file:
            # 如果提供了场景文件,从文件加载
            print(f"正在加载场景: {scene_file}")
            # 这里简化处理,实际可以用XML解析
            pass
            return

        # 解析密度参数
        if num_obstacles is None:
            if isinstance(density, str):
                mapping = {"low": 5, "mid": 10, "high": 20}
                num_obstacles = mapping.get(density, 10)
            elif isinstance(density, int):
                num_obstacles = density
            else:
                num_obstacles = 10

        print(f"创建简单障碍物环境... (num_obstacles={num_obstacles})")
        self._create_simple_obstacles(num_obstacles=num_obstacles)
    
    def _create_simple_obstacles(self, num_obstacles=10):
        """创建简单的障碍物环境

        num_obstacles: 障碍物数量
        """
        # 创建指定数量的随机位置的长方体障碍物
        for i in range(num_obstacles):
            # 随机大小
            half_extents = np.random.uniform(0.1, 0.5, 3)
            
            # 随机位置
            position = np.random.uniform(
                self.position_range[0], 
                self.position_range[1], 
                3
            )
            
            # 随机姿态
            orientation = p.getQuaternionFromEuler(
                np.random.uniform(0, 2*np.pi, 3)
            )
            
            # 创建障碍物
            box_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=half_extents
            )
            box_visual = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=half_extents,
                rgbaColor=[0.7, 0.7, 0.7, 1.0]
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_shape,
                baseVisualShapeIndex=box_visual,
                basePosition=position,
                baseOrientation=orientation
            )
            
            self.obstacle_ids.append(obstacle_id)
        
        print(f"创建了 {len(self.obstacle_ids)} 个障碍物")
    
    def create_sphere(self, position):
        """创建球体"""
        if self.sphere_id is not None:
            p.removeBody(self.sphere_id)
        
        sphere_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.sphere_radius
        )
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=[0.2, 0.8, 0.2, 0.7]
        )
        
        self.sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_shape,
            baseVisualShapeIndex=sphere_visual,
            basePosition=position
        )
        
        return self.sphere_id
    
    def create_obb(self, position, orientation):
        """创建OBB"""
        if self.obb_id is not None:
            p.removeBody(self.obb_id)
        
        box_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=self.obb_halfextents
        )
        box_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.obb_halfextents,
            rgbaColor=[0.2, 0.2, 0.8, 0.7]
        )
        
        self.obb_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_visual,
            basePosition=position,
            baseOrientation=orientation
        )
        
        return self.obb_id
    
    def check_collision(self, body_id):
        """检查给定物体与障碍物的碰撞"""
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getClosestPoints(
                bodyA=body_id,
                bodyB=obstacle_id,
                distance=0.0
            )
            
            # 如果有接触点且距离<=0,说明发生碰撞
            for point in contact_points:
                if point[8] <= 0:  # contact_distance
                    return True
        
        return False
    
    def random_position(self):
        """生成随机位置"""
        return np.random.uniform(
            self.position_range[0],
            self.position_range[1],
            3
        )
    
    def random_orientation(self):
        """生成随机方向(四元数)"""
        euler = np.random.uniform(0, 2*np.pi, 3)
        return p.getQuaternionFromEuler(euler)
    
    def quantize_position(self, position, num_bins=32):
        """量化位置坐标"""
        # 将连续空间离散化为哈希桶
        range_size = self.position_range[1] - self.position_range[0]
        bin_size = range_size / num_bins
        
        quantized = np.floor(
            (position - self.position_range[0]) / bin_size
        ).astype(int)
        
        # 限制在有效范围内
        quantized = np.clip(quantized, 0, num_bins - 1)
        
        return tuple(quantized)
    
    def quantize_orientation(self, quaternion, num_bins=16):
        """量化姿态(简化为欧拉角量化)"""
        euler = p.getEulerFromQuaternion(quaternion)
        
        # 将欧拉角离散化
        quantized = np.floor(
            np.array(euler) / (2*np.pi) * num_bins
        ).astype(int)
        
        quantized = quantized % num_bins
        
        return tuple(quantized)
    
    def train_cht(self, num_iterations=20000, num_bins_pos=32, num_bins_ori=16, 
                  training_ratio=0.8):
        """训练CHT并统计准确率和召回率"""
        num_training = int(num_iterations * training_ratio)
        num_testing = num_iterations - num_training
        
        print(f"\n开始训练CHT")
        print(f"  训练样本数: {num_training}")
        print(f"  测试样本数: {num_testing}")
        print("=" * 70)
        
        # 创建两个CHT策略,分别用于球体和OBB
        sphere_strategy = FixedThresholdStrategy(threshold=0.5, update_prob=1)
        obb_strategy = FixedThresholdStrategy(threshold=0.5, update_prob=1)
        
        # 收集数据
        sphere_training_data = []
        obb_training_data = []
        sphere_testing_data = []
        obb_testing_data = []
        
        # 训练阶段
        print("\n【训练阶段】")
        for iteration in range(num_training):
            if (iteration + 1) % 2000 == 0:
                print(f"训练进度: {iteration + 1}/{num_training}")
            
            # 1. 球体测试
            sphere_pos = self.random_position()
            self.create_sphere(sphere_pos)
            sphere_collision = self.check_collision(self.sphere_id)
            
            # 量化位置并生成哈希键
            sphere_key_pos = self.quantize_position(sphere_pos, num_bins_pos)
            # 直接使用tuple作为键,不需要hash函数
            
            # 训练阶段不需要预测,直接更新
            sphere_label = 0 if sphere_collision else 1
            sphere_strategy.update_history(sphere_key_pos, sphere_label)
            
            sphere_training_data.append({
                'position': sphere_pos,
                'collision': sphere_collision,
                'label': sphere_label
            })
            
            # 2. OBB测试
            obb_pos = self.random_position()
            obb_ori = self.random_orientation()
            self.create_obb(obb_pos, obb_ori)
            obb_collision = self.check_collision(self.obb_id)
            
            # 量化位置和姿态并生成哈希键
            obb_key_pos = self.quantize_position(obb_pos, num_bins_pos)
            obb_key_ori = self.quantize_orientation(obb_ori, num_bins_ori)
            # 合并位置和姿态作为键,直接使用tuple,不需要hash函数
            obb_hash_key = obb_key_pos
            
            # 训练阶段不需要预测,直接更新
            obb_label = 0 if obb_collision else 1
            obb_strategy.update_history(obb_hash_key, obb_label)
            
            obb_training_data.append({
                'position': obb_pos,
                'orientation': obb_ori,
                'collision': obb_collision,
                'label': obb_label
            })
        
        # 重置统计变量用于测试阶段
        sphere_strategy.all_zerozero = 0
        sphere_strategy.all_onezero = 0
        sphere_strategy.all_total_colliding = 0
        obb_strategy.all_zerozero = 0
        obb_strategy.all_onezero = 0
        obb_strategy.all_total_colliding = 0
        
        # 测试阶段
        print("\n【测试阶段】")
        for iteration in range(num_testing):
            if (iteration + 1) % 500 == 0:
                print(f"测试进度: {iteration + 1}/{num_testing}")
            
            # 1. 球体测试
            sphere_pos = self.random_position()
            self.create_sphere(sphere_pos)
            sphere_collision = self.check_collision(self.sphere_id)
            
            # 量化位置并生成哈希键
            sphere_key_pos = self.quantize_position(sphere_pos, num_bins_pos)
            # 直接使用tuple作为键
            
            # 预测碰撞
            sphere_prediction = sphere_strategy.predict_collision(sphere_key_pos)
            
            # 记录结果(测试阶段不更新历史)
            sphere_label = 0 if sphere_collision else 1
            
            sphere_testing_data.append({
                'position': sphere_pos,
                'collision': sphere_collision,
                'prediction': sphere_prediction,
                'label': sphere_label
            })
            
            # 2. OBB测试
            obb_pos = self.random_position()
            obb_ori = self.random_orientation()
            self.create_obb(obb_pos, obb_ori)
            obb_collision = self.check_collision(self.obb_id)
            
            # 量化位置和姿态并生成哈希键
            obb_key_pos = self.quantize_position(obb_pos, num_bins_pos)
            obb_key_ori = self.quantize_orientation(obb_ori, num_bins_ori)
            # 合并位置和姿态作为键
            obb_hash_key = obb_key_pos
            # 预测碰撞
            obb_prediction = obb_strategy.predict_collision(obb_hash_key)
            
            # 记录结果(测试阶段不更新历史)
            obb_label = 0 if obb_collision else 1
            
            obb_testing_data.append({
                'position': obb_pos,
                'orientation': obb_ori,
                'collision': obb_collision,
                'prediction': obb_prediction,
                'label': obb_label
            })
        
        print("\n训练完成!")
        print("=" * 70)
        
        # 计算统计指标(使用测试数据)
        self._compute_metrics(sphere_strategy, obb_strategy, 
                            sphere_testing_data, obb_testing_data,
                            sphere_training_data, obb_training_data)
        
        return (sphere_strategy, obb_strategy, 
                sphere_training_data, obb_training_data,
                sphere_testing_data, obb_testing_data)
    
    def _compute_metrics(self, sphere_strategy, obb_strategy, 
                        sphere_test_data, obb_test_data,
                        sphere_train_data, obb_train_data):
        """计算并打印准确率和召回率"""
        print("\n统计结果:")
        print("=" * 70)
        
        # 球体统计
        print("\n【球体碰撞预测】")
        print(f"  训练样本数: {len(sphere_train_data)}")
        print(f"  测试样本数: {len(sphere_test_data)}")
        
        # 训练集统计
        sphere_train_collisions = sum(1 for d in sphere_train_data if d['collision'])
        print(f"  训练集碰撞率: {(sphere_train_collisions/len(sphere_train_data))*100:.2f}%")
        
        # 测试集统计
        sphere_test_collisions = sum(1 for d in sphere_test_data if d['collision'])
        
        # 计算混淆矩阵
        true_positives = sum(
            1 for d in sphere_test_data 
            if d['collision'] and d['prediction']
        )
        false_positives = sum(
            1 for d in sphere_test_data 
            if not d['collision'] and d['prediction']
        )
        true_negatives = sum(
            1 for d in sphere_test_data 
            if not d['collision'] and not d['prediction']
        )
        false_negatives = sum(
            1 for d in sphere_test_data 
            if d['collision'] and not d['prediction']
        )
        
        # 计算指标
        sphere_accuracy = ((true_positives + true_negatives) / len(sphere_test_data)) * 100
        sphere_precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        sphere_recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        
        print(f"  测试集碰撞率: {(sphere_test_collisions/len(sphere_test_data))*100:.2f}%")
        print(f"  精确率 (Precision): {sphere_precision:.2f}%")
        print(f"  召回率 (Recall): {sphere_recall:.2f}%")
        print(f"  准确率 (Accuracy): {sphere_accuracy:.2f}%")
        print(f"  TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
        
        # OBB统计
        print("\n【OBB碰撞预测】")
        print(f"  训练样本数: {len(obb_train_data)}")
        print(f"  测试样本数: {len(obb_test_data)}")
        
        # 训练集统计
        obb_train_collisions = sum(1 for d in obb_train_data if d['collision'])
        print(f"  训练集碰撞率: {(obb_train_collisions/len(obb_train_data))*100:.2f}%")
        
        # 测试集统计
        obb_test_collisions = sum(1 for d in obb_test_data if d['collision'])
        
        # 计算混淆矩阵
        true_positives = sum(
            1 for d in obb_test_data 
            if d['collision'] and d['prediction']
        )
        false_positives = sum(
            1 for d in obb_test_data 
            if not d['collision'] and d['prediction']
        )
        true_negatives = sum(
            1 for d in obb_test_data 
            if not d['collision'] and not d['prediction']
        )
        false_negatives = sum(
            1 for d in obb_test_data 
            if d['collision'] and not d['prediction']
        )
        
        # 计算指标
        obb_accuracy = ((true_positives + true_negatives) / len(obb_test_data)) * 100
        obb_precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        obb_recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        
        print(f"  测试集碰撞率: {(obb_test_collisions/len(obb_test_data))*100:.2f}%")
        print(f"  精确率 (Precision): {obb_precision:.2f}%")
        print(f"  召回率 (Recall): {obb_recall:.2f}%")
        print(f"  准确率 (Accuracy): {obb_accuracy:.2f}%")
        print(f"  TP={true_positives}, FP={false_positives}, TN={true_negatives}, FN={false_negatives}")
        
        print("\n" + "=" * 70)
    
    def disconnect(self):
        """断开仿真连接"""
        p.disconnect(self.physics_client)


def main():
    """主函数"""
    print("=" * 70)
    print("简单的球体和OBB碰撞检测与CHT训练程序")
    print("=" * 70)
    # 我们将在三种密度下运行实验: low/mid/high
    densities = {"low": 5, "mid": 10, "high": 20}

    summary = []

    # 为每个密度创建独立的trainer并运行实验
    for name, n_obs in densities.items():
        print(f"\n=== 运行密度: {name} (obstacles={n_obs}) ===")
        trainer = SimpleCollisionCHTTrainer(use_gui=False)
        trainer.load_obstacles(density=name)

        results = trainer.train_cht(
            num_iterations=20000,
            num_bins_pos=32,
            num_bins_ori=16,
            training_ratio=0.8
        )

        sphere_strategy, obb_strategy = results[0], results[1]
        sphere_train_data, obb_train_data = results[2], results[3]
        sphere_test_data, obb_test_data = results[4], results[5]

        # 计算简要指标并加入汇总
        def compute_metrics_from_test(test_data):
            tp = sum(1 for d in test_data if d['collision'] and d['prediction'])
            fp = sum(1 for d in test_data if (not d['collision']) and d['prediction'])
            tn = sum(1 for d in test_data if (not d['collision']) and (not d['prediction']))
            fn = sum(1 for d in test_data if d['collision'] and (not d['prediction']))
            total = len(test_data)
            precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            accuracy = ((tp + tn) / total * 100) if total > 0 else 0
            coll_rate = (sum(1 for d in test_data if d['collision']) / total * 100) if total > 0 else 0
            return {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': precision, 'recall': recall, 'accuracy': accuracy, 'coll_rate': coll_rate,
                'total': total
            }

        sph_metrics = compute_metrics_from_test(sphere_test_data)
        obb_metrics = compute_metrics_from_test(obb_test_data)

        summary.append({
            'density': name,
            'n_obstacles': n_obs,
            'sphere': sph_metrics,
            'obb': obb_metrics
        })

        # 保存每个密度的原始结果文件
        outname = f"simple_cht_results_{name}.pkl"
        with open(outname, 'wb') as f:
            pickle.dump({
                'sphere_train_data': sphere_train_data,
                'sphere_test_data': sphere_test_data,
                'obb_train_data': obb_train_data,
                'obb_test_data': obb_test_data,
                'sphere_strategy': sphere_strategy.colldict,
                'obb_strategy': obb_strategy.colldict
            }, f)
        print(f"结果已保存到: {outname}")

        trainer.disconnect()

    # 打印汇总表
    print("\n=== 实验汇总 ===")
    for rec in summary:
        print(f"密度: {rec['density']} (obstacles={rec['n_obstacles']})")
        s = rec['sphere']
        o = rec['obb']
        print(f"  球体 -> Precision: {s['precision']:.2f}%, Recall: {s['recall']:.2f}%, Accuracy: {s['accuracy']:.2f}%, CollRate: {s['coll_rate']:.2f}% (N={s['total']})")
        print(f"  OBB   -> Precision: {o['precision']:.2f}%, Recall: {o['recall']:.2f}%, Accuracy: {o['accuracy']:.2f}%, CollRate: {o['coll_rate']:.2f}% (N={o['total']})")

    # 同时写入CSV
    try:
        import csv
        csvname = 'simple_cht_summary.csv'
        with open(csvname, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['density','n_obstacles','geom','precision','recall','accuracy','coll_rate','total','tp','fp','tn','fn'])
            for rec in summary:
                s = rec['sphere']
                o = rec['obb']
                writer.writerow([rec['density'], rec['n_obstacles'], 'sphere', f"{s['precision']:.2f}", f"{s['recall']:.2f}", f"{s['accuracy']:.2f}", f"{s['coll_rate']:.2f}", s['total'], s['tp'], s['fp'], s['tn'], s['fn']])
                writer.writerow([rec['density'], rec['n_obstacles'], 'obb', f"{o['precision']:.2f}", f"{o['recall']:.2f}", f"{o['accuracy']:.2f}", f"{o['coll_rate']:.2f}", o['total'], o['tp'], o['fp'], o['tn'], o['fn']])
        print(f"汇总已保存到: {csvname}")
    except Exception as e:
        print("保存CSV时出错:", e)

    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
