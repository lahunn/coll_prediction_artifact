# 图生成说明文档

fig1 : sphere_hashing_vs_link_hashing.png 比较精确率,召回率和计算代价

### 一键式流程（含说明）

1) 生成场景（障碍物）,批量生成碰撞数据
```bash
cd trace_generation/scripts
./launch_pred.sh
```
- 说明：输出放在 `trace_files/scene_benchmarks/<density>_rs/`；可通过 `--seed`、`--obb-vis`/`--sphere-vis` 控制行为。

2) 运行哈希策略评估（生成 CSV）
```bash
cd prediction_approaches/bash_script
./run_coord_cost_analysis.sh    # 生成 OBB 的评估 CSV
./run_sphere_cost_analysis.sh   # 生成 Sphere 的评估 CSV
```
- 输出：`result_files/coord_hashing_cost_results.csv` 与 `result_files/sphere_hashing_cost_results.csv`

3) 绘图并保存结果图像
```bash
cd prediction_approaches/plot
python plot_comparison_results.py
```
- 输出：`figs/` 下的对比图（precision/recall/cost 等）

---

fig2 :生成实际运动规划场景下的对比图

### 一键式流程（含说明）

1) 生成并分层问题集（GNN / 标准）
```bash
cd trace_generation/algorithm_evaluation
./generate_gnn_dataset.sh      # 使用 GNNMP 评估并按碰撞检查次数划分 G1..G5
./generate_standard_dataset.sh # 生成标准问题集（可选）
```

2) （可选）从 OBB/link 数据生成球体表示（用于 GNN 流程或需要 Link->Sphere 映射时）
```bash
# 批量生成（按 difficulty 和 id）
cd trace_generation/scripts
./generate_sphere_data.sh --algo bit_star --start 1 --end 100
```

3) 运行仿真对比并生成 per-difficulty CSV（link vs Sphere）
```bash
cd motion_planning_prediction/strategy_evaluation
./run_sphere_link_comparison.sh bit_star   # 或: ./run_sphere_link_comparison.sh gnnmp
```
- 说明：脚本会对 G1..G5、sphere_coord / link_coord 策略运行仿真并将结果写入 `result_files/`。

4) 绘图并保存 PNG（输出至 `plots/figs/`）
```bash
cd motion_planning_prediction/plots
python plot_cycle_comparison_sphere_link.py bit_star   # 或: python plot_cycle_comparison_sphere_link.py gnnmp
```

---

