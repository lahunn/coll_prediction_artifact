实验步骤说明:
1.随机场景随机pose,测试预测策略的精确率和召回率
  1.在trace_generation文件夹下,运行launch_pred.sh脚本,生成预测轨迹数据.
    launch_pred.sh脚本中可以设置不同的机器人,障碍物和场景参数.
  2.生成的数据会保存在../trace_files/scene_benchmarks/dens*_rs文件夹下.
  3.通过运行prediction_approaches/bash_script/run_sphere_cost_analysis.sh
    prediction_approaches/bash_script/run_coord_cost_analysis.sh
    来分析不同密度条件下,碰撞预测策略的效果.
  4.运行plot_comparison_results.py脚本,进行数据分析,绘制各类图表.

2.实际碰撞检测算法,测试预测策略的精确率和召回率
  1.调用trace_generation/bit_planning/generate_standard_dataset.sh脚本,
    生成标准数据集.
  2.运行trace_generation/sphere_as/generate_sphere_data.sh脚本,
    生成基于sphere_as的碰撞检测数据.
  3.运行motion_planning_prediction/test_sphere_obb_simulation.sh脚本,
    进行基于硬件结构的仿真测试