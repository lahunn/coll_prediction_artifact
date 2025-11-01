实验步骤说明:
1.在trace_generation文件夹下,运行launch_pred.sh脚本,生成预测轨迹数据.
  launch_pred.sh脚本中可以设置不同的机器人,障碍物和场景参数.
2.生成的数据会保存在../trace_files/scene_benchmarks/dens*_rs文件夹下.
3.通过运行prediction_approaches/bash_script/run_sphere_cost_analysis.sh
  prediction_approaches/bash_script/run_coord_cost_analysis.sh
  来分析不同密度条件下,碰撞预测策略的效果.
