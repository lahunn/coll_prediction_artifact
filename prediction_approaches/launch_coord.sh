rm -rf result_files/coord_*.csv

echo   "密度,量化位数,半径量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 -1      1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 2       1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 1       1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 0.5     1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 0.125   1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 0.03125 1 >> result_files/coord_sphere_low.csv
python coord_hashing_sphere.py    low  4 2 0       1 >> result_files/coord_sphere_low.csv

echo   "密度,量化位数,半径量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 -1      1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 2       1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 1       1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 0.5     1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 0.125   1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 0.03125 1 >> result_files/coord_sphere_mid.csv
python coord_hashing_sphere.py    mid  4 2 0       1 >> result_files/coord_sphere_mid.csv

echo   "密度,量化位数,半径量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 -1      1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 2       1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 1       1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 0.5     1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 0.125   1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 0.03125 1 >> result_files/coord_sphere_high.csv
python coord_hashing_sphere.py    high 4 2 0       1 >> result_files/coord_sphere_high.csv

echo   "密度,量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_low.csv
python coord_hashing.py           low  4 -1      1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 2       1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 1       1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 0.5     1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 0.125   1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 0.03125 1 >> result_files/coord_low.csv
python coord_hashing.py           low  4 0       1 >> result_files/coord_low.csv

echo   "密度,量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_mid.csv
python coord_hashing.py           mid  4 -1      1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 2       1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 1       1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 0.5     1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 0.125   1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 0.03125 1 >> result_files/coord_mid.csv
python coord_hashing.py           mid  4 0       1 >> result_files/coord_mid.csv

echo   "密度,量化位数,碰撞阈值,采样率,精确率,召回率" >>   result_files/coord_high.csv
python coord_hashing.py           high 4 -1      1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 2       1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 1       1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 0.5     1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 0.125   1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 0.03125 1 >> result_files/coord_high.csv
python coord_hashing.py           high 4 0       1 >> result_files/coord_high.csv