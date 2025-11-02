


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# ---------------------------
# 函数：加载文本文件数据
# ---------------------------
def load_data(file_path):
    """
    按行加载文件数据，每行以空格分隔为浮点数列表，返回嵌套列表
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                float_values = [float(value) for value in line.split()]
                data.append(float_values)
    return data

# ---------------------------
# 方法：对加载的列表数据进行填充，构造统一大小的 numpy 数组
# ---------------------------
def pad_data(data_list):
    num_rows = len(data_list)
    max_length = max(len(row) for row in data_list)
    pad = np.zeros((num_rows, max_length), dtype=np.float32)
    for i, row in enumerate(data_list):
        pad[i, :len(row)] = np.array(row, dtype=np.float32)
    return pad

# ---------------------------
# 方法：仅转换数据，不进行插值填充
# ---------------------------
def convert_data(data):
    """
    对数据中大于 0.1 的值进行转换，即 data = 337.5 / data，
    对于 0 值保持不变
    """
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    rows, cols = data.shape
    new_data = data.copy()
    for i in range(rows):
        for j in range(cols):
            # 统计区间数量
            if new_data[i, j] < 5.0 and new_data[i, j] > 1.0:
                count0 += 1
            if new_data[i, j] > 5.0 and new_data[i, j] < 6.5:
                count1 += 1
            if new_data[i, j] > 6.5 and new_data[i, j] < 7:
                count2 += 1
            if new_data[i, j] > 7.0 and new_data[i, j] < 8.0:
                count3 += 1

            if new_data[i, j] > 0.1:
                new_data[i, j] = 337.5 / new_data[i, j]
    print('计数')
    print(f'{count0} {count1} {count2} {count3}')
    return new_data

# ---------------------------
# 方法：插值函数，填充数据中为 0 的缺失位置
# ---------------------------
def Find_vacancy_insert(data, l):
    """
    data: 输入的二维数组（每行代表一个样本）
    l: 插值时考虑的左右边界宽度
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
            elif data_ext[i, j] == 0 and vacant_number != 0:
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
        if len(valid_idx) > 0 and len(nan_idx) > 0:
            f = interp1d(valid_idx, row[valid_idx], kind='linear', fill_value="extrapolate")
            data_ext[i, nan_idx] = f(nan_idx)
    return data_ext[:, l:-l]

# ---------------------------
# 公用函数：生成同心圆图的点坐标
# ---------------------------
def get_circle_coords(data, max_radius=5):
    """
    根据同心圆可视化要求，返回 x_all, y_all 坐标以及数据值的展开数组。
    参数：
      data: 二维数据，内部先进行转置，每行代表一个圆环上的采样值；
      max_radius: 图中最大的半径值，用于计算半径步长。
    """
    data_t = data.T  # 转置后：每行代表一个圆环
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

# ---------------------------
# 函数：同心圆格式可视化（基础版）
# ---------------------------
def visualize_circle(data, title="", max_radius=5):
    """
    将传入二维矩阵 data 转换为同心圆热力图进行显示。
    """
    x_all, y_all, c_all = get_circle_coords(data, max_radius=max_radius)
    norm = Normalize(vmin=35, vmax=50)
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x_all, y_all, c=c_all, cmap='viridis', norm=norm)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.colorbar(sc)
    plt.show()

# ---------------------------
# 新增函数：利用Kasa方法对点集拟合圆
# ---------------------------
def fit_circle(x, y):
    """
    使用Kasa最小二乘法拟合圆
    方程形式：x^2 + y^2 + Ax + By + C = 0
    返回：
      center: (x0, y0)
      r: 半径
    """
    # 构建矩阵 D 和向量 f，其中 f = -(x^2 + y^2)
    D = np.column_stack((x, y, np.ones_like(x)))
    f = -(x**2 + y**2)
    # 求解最小二乘问题
    params, _, _, _ = np.linalg.lstsq(D, f, rcond=None)
    A, B, C = params
    # 计算圆心和半径
    x0 = -A / 2
    y0 = -B / 2
    r = np.sqrt((A/2)**2 + (B/2)**2 - C)
    return (x0, y0), r

# ---------------------------
# 新增函数：在 file1 的同心圆图中根据差值情况标注出差值绝对值 < threshold 且半径在 [rmin, rmax] 的点，
# 并利用部分点拟合近似圆形，显示其半径和面积
# ---------------------------
def visualize_file1_with_diff_mark(file1_data, diff, threshold=0.1, rmin=7, rmax=25, title="File1: Marked & Fitted Circle"):
    """
    file1_data: file1 经过去噪后得到的二维数据，与 diff 尺寸一致。
    diff: file1 与 file0 数据的差值数组。
    threshold: 标记条件，当 |diff| < threshold 时标记该点。
    rmin, rmax: 标记点的极径取值区间（单位与图中坐标一致）。
    拟合时选取标记出来的点，拟合出一个近似圆，并计算该圆的半径和面积。
    """
    # 此处设置 max_radius 为 30，使得图中半径范围足够覆盖 0~30
    max_radius = 30
    x_all, y_all, c_all = get_circle_coords(file1_data, max_radius=max_radius)
    # 将 diff 数组展平
    diff_flat = diff.T.reshape(-1)
    # 计算每个点的极径
    r_all = np.sqrt(x_all**2 + y_all**2)
    
    norm = Normalize(vmin=35, vmax=50)
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x_all, y_all, c=c_all, cmap='viridis', norm=norm)
    
    # 筛选条件：|diff| < threshold 且极径在 [rmin, rmax]
    mask = (np.abs(diff_flat) < threshold) & (r_all >= rmin) & (r_all <= rmax)
    
    # 标记符合条件的点
    plt.scatter(x_all[mask], y_all[mask], facecolors='none',
                edgecolors='r', s=40, linewidths=1.5,
                label=f'|diff|<{threshold} & r in [{rmin},{rmax}]')
    
    # 检查是否有足够的点进行拟合
    if np.sum(mask) >= 3:
        fit_x = x_all[mask]
        fit_y = y_all[mask]
        center, radius = fit_circle(fit_x, fit_y)
        area = np.pi * (radius ** 2)
        # 生成圆上的点用于绘制拟合圆弧
        theta_fit = np.linspace(0, 2*np.pi, 200)
        x_fit = center[0] + radius * np.cos(theta_fit)
        y_fit = center[1] + radius * np.sin(theta_fit)
        plt.plot(x_fit, y_fit, 'b--', linewidth=2, label=f'Fit circle: r={radius:.2f}, area={area:.2f}')
        # 在图中标注圆心
        plt.plot(center[0], center[1], 'bo', markersize=5)
    else:
        print("用于拟合圆的点数量不足！")
    
    plt.legend()
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.colorbar(sc)
    plt.show()

# ---------------------------
# 方法：对两个时段数据进行处理，并求差
# ---------------------------
def process_two_files(file0, file1, l=100, sigma=1.0):
    """
    对两个文件原始数据分别进行转换、插值、去噪处理，
    返回处理后的数据及其差值（file1 - file0）。
    """
    # 加载数据，转换为列表
    data_list0 = load_data(file0)
    data_list1 = load_data(file1)

    # 填充数据，将列表转为统一大小的数组
    pad0 = pad_data(data_list0)
    pad1 = pad_data(data_list1)

    # 转换数据
    conv0 = convert_data(pad0)
    conv1 = convert_data(pad1)

    # 插值处理（先转置、插值、再转置回来）
    data0_T = conv0.T
    data1_T = conv1.T
    interp0 = Find_vacancy_insert(data0_T, l).T
    interp1 = Find_vacancy_insert(data1_T, l).T

    # 去噪（高斯滤波）
    denoise0 = gaussian_filter(interp0, sigma=sigma)
    denoise1 = gaussian_filter(interp1, sigma=sigma)

    # 差值（file1 - file0），要求两者尺寸一致
    diff = denoise1 - denoise0
    return denoise0, denoise1, diff

# ---------------------------
# 函数：差值数据可视化（同心圆格式）
# ---------------------------
def visualize_difference(diff, title="Difference (File1 - File0)", max_radius=30):
    x_all, y_all, c_all = get_circle_coords(diff, max_radius=max_radius)
    norm = Normalize(vmin=np.min(c_all), vmax=np.max(c_all))
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x_all, y_all, c=c_all, cmap='coolwarm', norm=norm)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.xlim(-max_radius, max_radius)
    plt.ylim(-max_radius, max_radius)
    plt.colorbar(sc)
    plt.savefig("/Users/txh/Desktop/medical_segment/output.png", dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------
# 主程序：处理样本，
# 分别显示 file0、file1 的去噪图、差值图，
# 并在 file1 图中标注出差值绝对值 < 0.1 且半径在 [7,25] 的点，
# 对这些点进行圆形拟合，显示拟合得到的圆的半径和面积。
# ---------------------------
def main():
    # 请根据实际情况修改路径
    input_root = '/Users/txh/Desktop/after_filtering_segmentation_6.0'
    
    # 获取 input_root 下所有子文件夹，并排序
    subfolders = [os.path.join(input_root, d) for d in os.listdir(input_root)
                  if os.path.isdir(os.path.join(input_root, d))]
    subfolders.sort()

    if len(subfolders) < 1:
        print("样本文件夹不足！")
        return

    sample_folder = subfolders[0]
    print(f"处理样本: {sample_folder}")
    
    # 在样本文件夹中选择排序后的文件，要求至少两个文件（file0 与 file1）
    files = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder)
             if os.path.isfile(os.path.join(sample_folder, f))]
    files.sort()

    if len(files) < 2:
        print("样本中至少需要两个文件！")
        return

    file0 = files[0]
    file1 = files[1]
    print(f"处理文件0: {file0}")
    print(f"处理文件1: {file1}")
    
    # 对 file0 与 file1 分别进行转换、插值、去噪，并计算差值
    processed0, processed1, diff = process_two_files(file0, file1, l=100, sigma=1.0)
    
    # 可视化各阶段结果：
    visualize_circle(processed0, title="File0: After Denoising (Gaussian Filter)", max_radius=30)
    visualize_circle(processed1, title="File1: After Denoising (Gaussian Filter)", max_radius=30)
    visualize_difference(diff, title="Difference (File1 - File0)", max_radius=30)
    
    # 在 file1 图中标注出差值绝对值 <0.1 且半径在 [7,25] 的点，并对其拟合近似圆形
    visualize_file1_with_diff_mark(processed1, diff, threshold=0.1, rmin=7, rmax=15,
                                   title="File1: Marked Points & Fitted Circle")
    
if __name__ == "__main__":
    main()