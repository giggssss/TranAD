import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

# plt.style.use(['science', 'ieee'])
plt.style.use(['default'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def to_numpy(x):
    """텐서인지 넘파이 배열인지 확인하고 넘파이 배열로 변환"""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

def plotter(name, y_true, y_pred, ascore, labels):
    dimension_names = ['initDegreeX', 'initDegreeY', 'initDegreeZ',
                      'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
                      'initCrack', 'crackAmount', 'temperature', 'humidity']
    
    if 'TranAD' in name:
        y_true = torch.roll(y_true, 1, 0)
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    for dim in range(y_true.shape[1]):
        if dim >= len(dimension_names):
            dim_name = f'Dimension {dim}'
        else:
            dim_name = dimension_names[dim]
        
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
        
        # 텐서 또는 넘파이 배열을 넘파이 배열로 변환
        y_t = to_numpy(y_t)
        y_p = to_numpy(y_p)
        l = to_numpy(l)
        a_s = to_numpy(a_s)
        
        # 스무딩 적용
        y_t_smooth = smooth(y_t)
        y_p_smooth = smooth(y_p)
        a_s_smooth = smooth(a_s)
        
        # 임계값 계산 (평균 + 3*표준편차)
        threshold = np.mean(a_s_smooth) + 3 * np.std(a_s_smooth)
        
        # 임계값을 초과하는 지점 찾기
        anomalies = np.where(a_s_smooth > threshold)[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension{dim} = {dim_name}')
        
        # 실제 값과 예측 값 플롯
        ax1.plot(y_t_smooth, linewidth=0.2, label='True')
        ax1.plot(y_p_smooth, '-', alpha=0.6, linewidth=0.3, label='Predicted')
        
        anomalies_indices = np.where(l == 1)[0]
        ax1.scatter(anomalies_indices, y_t_smooth[anomalies_indices], color='b', s=1, label='Detected Anomaly')
        
        
        # 레이블 플롯 (이상치 표시)
        ax3 = ax1.twinx()
        anomalies_TP = np.intersect1d(anomalies_indices, anomalies)
        # ax3.fill_between(np.arange(anomalies_TP.shape[0]), anomalies_TP, color='blue', alpha=0.2)
        # ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        # ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.2)
        
        # if dim == 0:
            # ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        
        # 이상치 점수 플롯
        ax2.plot(a_s_smooth, linewidth=0.2, color='g', label='Anomaly Score')
        ax2.axhline(y=threshold, color='r', linestyle='--', label='Threshold', linewidth=0.2)
        ax2.scatter(anomalies, a_s_smooth[anomalies], color='r', s=1, label='Detected Anomaly')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        # ax2.legend(loc='upper right', fontsize='small')
        
        pdf.savefig(fig)
        plt.close()
    pdf.close()
