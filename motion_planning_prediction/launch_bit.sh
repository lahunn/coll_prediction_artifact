rm -rf result_files/*.csv
python prediction_simulation_2D.py 1 1 64 MPNET | tee logfile result_files/mpnet_2d_pred.csv 
python prediction_simulation_2D.py 1 1 64 BIT | tee logfile result_files/bit_2d_pred.csv 
python prediction_simulation_2D.py 1 1 64 GNN | tee logfile result_files/gnn_2d_pred.csv 

python CSP_simulation_2D.py MPNET | tee logfile result_files/mpnet_2d_csp.csv 
python CSP_simulation_2D.py BIT | tee logfile result_files/bit_2d_csp.csv 
python CSP_simulation_2D.py GNN | tee logfile result_files/gnn_2d_csp.csv 

python prediction_simulation_nDOF.py 1 0.125 8 MPNET | tee logfile result_files/mpnet_nDOF_pred.csv 
python prediction_simulation_nDOF.py 1 0.125 8 BIT | tee logfile result_files/bit_nDOF_pred.csv 
python prediction_simulation_nDOF.py 1 0.125 8 GNN | tee logfile result_files/gnn_nDOF_pred.csv 

python CSP_simulation_nDOF.py MPNET | tee logfile result_files/mpnet_nDOF_csp.csv 
python CSP_simulation_nDOF.py BIT | tee logfile result_files/bit_nDOF_csp.csv 
python CSP_simulation_nDOF.py GNN | tee logfile result_files/gnn_nDOF_csp.csv 

# 切换到结果文件目录
cd result_files

# === 合并2D场景的结果文件 ===
# 使用paste命令将CSP和预测结果按空格分隔合并到一个文件中
# 合并GNN模型的2D CSP和预测结果
paste -d " " gnn_2d_csp.csv gnn_2d_pred.csv > gnn_2d.csv
# 合并BIT*模型的2D CSP和预测结果
paste -d " " bit_2d_csp.csv bit_2d_pred.csv > bit_2d.csv
# 合并MPNet模型的2D CSP和预测结果
paste -d " " mpnet_2d_csp.csv mpnet_2d_pred.csv > mpnet_2d.csv
# === 合并nDOF场景的结果文件 ===
# 合并GNN模型的nDOF CSP和预测结果
paste -d " " gnn_nDOF_csp.csv gnn_nDOF_pred.csv > gnn_7d.csv
# 合并BIT*模型的nDOF CSP和预测结果
paste -d " " bit_nDOF_csp.csv bit_nDOF_pred.csv > bit_7d.csv
# 合并MPNet模型的nDOF CSP和预测结果
paste -d " " mpnet_nDOF_csp.csv mpnet_nDOF_pred.csv > mpnet_7d.csv
cd ../



