#!/bin/bash
# ========================================
# 运动规划碰撞预测性能评估脚本
# ========================================
# 功能: 评估不同运动规划算法(MPNet/BIT*/GNN)在2D和nDOF场景下的碰撞预测性能
# 对比: CSP(基线方法) vs Prediction(预测优化方法)
# ========================================

# === 清理旧的结果文件 ===
rm -rf result_files/*.csv

# ========================================
# 第一部分: 2D场景预测方法评估
# ========================================
# 参数说明: <阈值> <采样率> <队列长度> <算法类型>
# - 阈值=1: 碰撞预测阈值(colldict[key][0] > threshold * colldict[key][1])
# - 采样率=1: 自由样本采样率(1.0表示全部采样)
# - 队列长度=64: 非碰撞队列最大长度(qnoncoll_len)
# - 算法类型: MPNET/BIT/GNN

# 运行MPNet算法的2D预测评估
python prediction_simulation_2D.py 1 1 64 MPNET | tee logfile result_files/mpnet_2d_pred.csv 
# 运行BIT*算法的2D预测评估
python prediction_simulation_2D.py 1 1 64 BIT | tee logfile result_files/bit_2d_pred.csv 
# 运行GNN算法的2D预测评估
python prediction_simulation_2D.py 1 1 64 GNN | tee logfile result_files/gnn_2d_pred.csv 

# ========================================
# 第二部分: 2D场景CSP基线方法评估
# ========================================
# 参数说明: <算法类型>
# CSP(Collision Space Partitioning): 使用静态启发式顺序,不使用预测

# 运行MPNet算法的2D CSP基线评估
python CSP_simulation_2D.py MPNET | tee logfile result_files/mpnet_2d_csp.csv 
# 运行BIT*算法的2D CSP基线评估
python CSP_simulation_2D.py BIT | tee logfile result_files/bit_2d_csp.csv 
# 运行GNN算法的2D CSP基线评估
python CSP_simulation_2D.py GNN | tee logfile result_files/gnn_2d_csp.csv 

# ========================================
# 第三部分: nDOF场景(7自由度)预测方法评估
# ========================================
# 参数说明: <阈值> <采样率> <队列倍数> <算法类型>
# - 阈值=1: 碰撞预测阈值
# - 采样率=0.125: 自由样本采样率(1/8采样,减少内存开销)
# - 队列倍数=8: qnoncoll_len = 7 * 8 = 56 (适配7自由度机器人)
# - 算法类型: MPNET/BIT/GNN

# 运行MPNet算法的nDOF预测评估
python prediction_simulation_nDOF.py 1 0.125 8 MPNET | tee logfile result_files/mpnet_nDOF_pred.csv 
# 运行BIT*算法的nDOF预测评估
python prediction_simulation_nDOF.py 1 0.125 8 BIT | tee logfile result_files/bit_nDOF_pred.csv 
# 运行GNN算法的nDOF预测评估
python prediction_simulation_nDOF.py 1 0.125 8 GNN | tee logfile result_files/gnn_nDOF_pred.csv 

# ========================================
# 第四部分: nDOF场景CSP基线方法评估
# ========================================
# 参数说明: <算法类型>

# 运行MPNet算法的nDOF CSP基线评估
python CSP_simulation_nDOF.py MPNET | tee logfile result_files/mpnet_nDOF_csp.csv 
# 运行BIT*算法的nDOF CSP基线评估
python CSP_simulation_nDOF.py BIT | tee logfile result_files/bit_nDOF_csp.csv 
# 运行GNN算法的nDOF CSP基线评估
python CSP_simulation_nDOF.py GNN | tee logfile result_files/gnn_nDOF_csp.csv 

# ========================================
# 第五部分: 结果文件合并处理
# ========================================
# 目的: 将CSP基线和Prediction方法的结果合并,便于对比分析
# 合并格式: <CSP查询数> <Oracle查询数> <Prediction查询数> <Oracle查询数>

# 切换到结果文件目录
cd result_files

# === 合并2D场景的结果文件 ===
# 使用paste命令将CSP和预测结果按空格分隔合并到一个文件中
# 输出格式: 每行包含同一个benchmark的CSP和Prediction性能数据

# 合并GNN模型的2D CSP和预测结果
paste -d " " gnn_2d_csp.csv gnn_2d_pred.csv > gnn_2d.csv
# 合并BIT*模型的2D CSP和预测结果
paste -d " " bit_2d_csp.csv bit_2d_pred.csv > bit_2d.csv
# 合并MPNet模型的2D CSP和预测结果
paste -d " " mpnet_2d_csp.csv mpnet_2d_pred.csv > mpnet_2d.csv

# === 合并nDOF场景的结果文件 ===
# 输出文件命名为*_7d.csv,表示7自由度(7-DOF)机器人场景

# 合并GNN模型的nDOF CSP和预测结果
paste -d " " gnn_nDOF_csp.csv gnn_nDOF_pred.csv > gnn_7d.csv
# 合并BIT*模型的nDOF CSP和预测结果
paste -d " " bit_nDOF_csp.csv bit_nDOF_pred.csv > bit_7d.csv
# 合并MPNet模型的nDOF CSP和预测结果
paste -d " " mpnet_nDOF_csp.csv mpnet_nDOF_pred.csv > mpnet_7d.csv

# 返回上级目录
cd ../

# ========================================
# 执行完成
# ========================================
# 生成的合并文件可用于:
# 1. plot_fig15.py - 绘制不同复杂度下的性能对比图
# 2. plot_fig16.py - 绘制性能/功耗/面积分析图
# ========================================



