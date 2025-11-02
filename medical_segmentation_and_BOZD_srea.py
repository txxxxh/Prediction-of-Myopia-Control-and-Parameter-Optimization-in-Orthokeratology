


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------
# 基本数据加载与预处理函数（matrix_list2中的数据已处理好）
# --------------------------------------------------------------
def load_data(file_path):
    """
    按行加载文本文件数据，每行以空格分隔为浮点数列表
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                float_values = [float(value) for value in line.split()]
                data.append(float_values)
    return data

def pad_data(data_list):
    """
    使用零填充构造统一大小的 numpy 数组
    """
    num_rows = len(data_list)
    max_length = max(len(row) for row in data_list)
    pad = np.zeros((num_rows, max_length), dtype=np.float32)
    for i, row in enumerate(data_list):
        pad[i, :len(row)] = np.array(row, dtype=np.float32)
    return pad

def convert_data(data):
    """
    对数据中大于0.1的值进行转换（data = 337.5 / data），0值保持不变
    """
    rows, cols = data.shape
    new_data = data.copy()
    for i in range(rows):
        for j in range(cols):
            if new_data[i, j] > 0.1:
                new_data[i, j] = 337.5 / new_data[i, j]
    return new_data

def Find_vacancy_insert(data, l):
    """
    插值填充缺失数据（值为0的位置）
    """
    data_tail = data[:, -l:]
    data_head = data[:, 0:l]
    data_ext = np.concatenate((data_tail, data, data_head), axis=1)
    n_rows = data_ext.shape[0]
    for i in range(n_rows):
        tmp1 = -1
        vacant_number = 0
        for j in range(len(data_ext[i])):
            if data_ext[i, j] == 0 and vacant_number == 0:
                tmp1 = j
                vacant_number += 1
            elif data_ext[i, j] == 0:
                vacant_number += 1
            elif data_ext[i, j] != 0 and (1 <= vacant_number <= l):
                data_ext[i, tmp1:j] = np.nan
                vacant_number = 0
            elif data_ext[i, j] != 0 and vacant_number > l:
                vacant_number = 0
        if 1 <= vacant_number <= l:
            data_ext[i, tmp1:] = np.nan
    for i in range(n_rows):
        row = data_ext[i]
        nan_idx = np.where(np.isnan(row))[0]
        valid_idx = np.where(~np.isnan(row))[0]
        if valid_idx.size > 0 and nan_idx.size > 0:
            f = interp1d(valid_idx, row[valid_idx], kind='linear', fill_value="extrapolate")
            data_ext[i, nan_idx] = f(nan_idx)
    return data_ext[:, l:-l]

# --------------------------------------------------------------
# 工具函数：将2D数据转换为同心圆坐标
# --------------------------------------------------------------
def get_circle_coords(data, max_radius=5):
    """
    将二维矩阵 data 转换为同心圆坐标，返回 x_all, y_all, c_all（c_all 为数据展开后的值）
    参数：
      data: 每行代表一个圆环的采样值；
      max_radius: 图中最大半径，用于确定采样步长
    """
    data_t = data.T
    n_rings = data_t.shape[0]
    radius_step = max_radius / n_rings
    x_list, y_list = [], []
    for i in range(n_rings):
        theta = np.linspace(0, 2 * np.pi, data_t.shape[1], endpoint=False)
        r = i * radius_step
        x_list.append(r * np.cos(theta))
        y_list.append(r * np.sin(theta))
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    c_all = np.concatenate(list(data_t))
    return x_all, y_all, c_all

# --------------------------------------------------------------
# 可视化函数——将数据以同心圆格式显示（补充此函数以避免未定义错误）
# --------------------------------------------------------------
def visualize_circle(data, title="", max_radius=5):
    """
    将二维矩阵 data 转换为同心圆图并显示
    """
    x_all, y_all, c_all = get_circle_coords(data, max_radius=max_radius)
    plt.figure(figsize=(8,8))
    sc = plt.scatter(x_all, y_all, c=c_all, cmap='viridis')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.colorbar(sc)
    plt.show()

# --------------------------------------------------------------
# 圆拟合函数：采用 Kasa 方法拟合圆（不剔除点）
# --------------------------------------------------------------
def fit_circle(x, y):
    """
    使用 Kasa 最小二乘法拟合圆，假定圆方程：
         x^2 + y^2 + A*x + B*y + C = 0
    返回：(圆心, 半径)
    """
    D = np.column_stack((x, y, np.ones_like(x)))
    f = -(x**2 + y**2)
    params, _, _, _ = np.linalg.lstsq(D, f, rcond=None)
    A, B, C = params
    x0 = -A / 2
    y0 = -B / 2
    # 取绝对值确保半径为正
    r = np.sqrt(abs((A/2)**2 + (B/2)**2 - C))
    return (x0, y0), r

# --------------------------------------------------------------
# 基于差值拟合圆
# 对于非基准阶段数据，先与第一阶段数据(基准)做差，
# 然后从 stage 数据的同心圆坐标中筛选出
# 满足 |diff| < threshold 且 极径在 [rmin, rmax] 内的点，
# 用这些点直接拟合圆形
# --------------------------------------------------------------
def compute_fitted_circle_from_diff(baseline, stage, threshold=0.1, rmin=7, rmax=15, max_radius_display=30):
    """
    参数：
      baseline: 第一阶段处理后的矩阵（基准）
      stage: 当前阶段处理后的矩阵
      threshold: 差值阈值（仅保留 |stage - baseline| < threshold 的点）
      rmin, rmax: 筛选点时要求的极径范围
      max_radius_display: 用于坐标转换的显示最大半径
    返回：(圆心, 半径, 面积)；若点数不足或拟合结果不符合条件，则返回 None
    """
    # 计算差值矩阵（假设两矩阵尺寸一致）
    diff = stage - baseline
    # 将 stage 矩阵转换为同心圆坐标
    x_all, y_all, _ = get_circle_coords(stage, max_radius=max_radius_display)
    # 将 diff 矩阵展平
    diff_flat = diff.T.reshape(-1)
    # 每个点的极径
    r_all = np.sqrt(x_all**2 + y_all**2)
    # 筛选条件：|diff| < threshold 且极径在 [rmin, rmax] 内
    mask = (np.abs(diff_flat) < threshold) & (r_all >= rmin) & (r_all <= rmax)
    if np.sum(mask) < 3:
        return None
    fit_x = x_all[mask]
    fit_y = y_all[mask]
    center, radius = fit_circle(fit_x, fit_y)
    if rmin <= radius <= rmax:
        area = np.pi * (radius ** 2)
        return center, radius, area
    else:
        return None

# --------------------------------------------------------------
# 辅助函数：利用 IQR 方法剔除离群值（用于计算平均值时）
# --------------------------------------------------------------
def get_outlier_mask(values):
    values = np.array(values)
    if len(values) < 3:
        return np.ones(len(values), dtype=bool)
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (values >= lower_bound) & (values <= upper_bound)
    return mask

# --------------------------------------------------------------
# 主程序：对每个人（子文件夹）不同阶段数据，
# 将除第一阶段（基准）外，各阶段数据与基准数据作差，
# 根据差值条件拟合圆形（获取半径和面积），
# 对所有有效阶段利用 IQR 剔除离群值后求平均，
# 最后写入 Excel 文件
# --------------------------------------------------------------
def main_for_excel():
    input_matrix_dir = '/Users/txh/Desktop/research/lwk/ok/Data_processed/tgl_after_filtering'
    excel_out_path = '/Users/txh/Desktop/segment_excel.xlsx'
    
    # 每个人数据存放在各自子文件夹中
    persons = [os.path.join(input_matrix_dir, d) for d in os.listdir(input_matrix_dir)
               if os.path.isdir(os.path.join(input_matrix_dir, d))]
    persons.sort()
    
    records = []
    for person_dir in persons:
        person_id = os.path.basename(person_dir)
        print(f"处理人: {person_id}")
        # 读取该人所有txt文件（各阶段矩阵），按文件名排序后取前5个
        files = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                 if os.path.isfile(os.path.join(person_dir, f)) and f.lower().endswith('.tgl')]
        files.sort()
        if len(files) < 1:
            print(f"  {person_id} 无数据文件")
            continue
        
        # 加载第一阶段作为基准
        try:
            baseline = np.loadtxt(files[0])
        except Exception as e:
            print(f"读取基准文件 {files[0]} 失败: {e}")
            continue

        # 对每个阶段（从第二个开始）根据基准做差拟合圆
        stage_radii = [None] * 5
        stage_areas  = [None] * 5
        # 第一阶段视为基准，不进行拟合（可留空或记录基准参数）
        stage_radii[0] = None
        stage_areas[0]  = None
        
        for idx, file_path in enumerate(files[1:5], start=1):
            try:
                stage = np.loadtxt(file_path)
            except Exception as e:
                print(f"读取 {file_path} 失败: {e}")
                continue
            result = compute_fitted_circle_from_diff(baseline, stage, threshold=0.1, rmin=7, rmax=15, max_radius_display=30)
            if result:
                center, radius, area = result
                stage_radii[idx] = radius
                stage_areas[idx]  = area
                print(f"  阶段 {idx+1}: radius = {radius:.2f}, area = {area:.2f}")
            else:
                print(f"  阶段 {idx+1}: 无满足条件的拟合圆")
        
        # 统计不包括基准阶段的有效数据求平均（采用 IQR 离群值剔除）
        valid_radii = [r for r in stage_radii if r is not None]
        valid_areas  = [a for a in stage_areas if a is not None]
        if valid_radii:
            mask_r = get_outlier_mask(valid_radii)
            filtered_radii = np.array(valid_radii)[mask_r]
            avg_radius = float(np.mean(filtered_radii)) if len(filtered_radii) > 0 else float(np.mean(valid_radii))
        else:
            avg_radius = None
        if valid_areas:
            mask_a = get_outlier_mask(valid_areas)
            filtered_areas = np.array(valid_areas)[mask_a]
            avg_area = float(np.mean(filtered_areas)) if len(filtered_areas) > 0 else float(np.mean(valid_areas))
        else:
            avg_area = None
        
        record = {
            "Person": person_id,
            "Stage1_Radius": stage_radii[0],
            "Stage1_Area": stage_areas[0],
            "Stage2_Radius": stage_radii[1],
            "Stage2_Area": stage_areas[1],
            "Stage3_Radius": stage_radii[2],
            "Stage3_Area": stage_areas[2],
            "Stage4_Radius": stage_radii[3],
            "Stage4_Area": stage_areas[3],
            "Stage5_Radius": stage_radii[4],
            "Stage5_Area": stage_areas[4],
            "Avg_Radius": avg_radius,
            "Avg_Area": avg_area
        }
        records.append(record)
    
    # 写入 Excel 文件
    df = pd.DataFrame(records, columns=[
        "Person",
        "Stage1_Radius", "Stage1_Area",
        "Stage2_Radius", "Stage2_Area",
        "Stage3_Radius", "Stage3_Area",
        "Stage4_Radius", "Stage4_Area",
        "Stage5_Radius", "Stage5_Area",
        "Avg_Radius", "Avg_Area"
    ])
    df.to_excel(excel_out_path, index=False)
    print(f"结果已写入: {excel_out_path}")

if __name__ == "__main__":
    main_for_excel()