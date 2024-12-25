import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# train.npy 파일에서 데이터 로드
data = np.load('/workspace/processed/Gyeongsan/train.npy', allow_pickle=True)
num_columns = data.shape[1]
with PdfPages('all_columns_visualization_train.pdf') as pdf:
    for i in range(num_columns):
        plt.figure()
        plt.plot(data[:, i])
        plt.title(f'Column {i+1} Visualization')
        plt.xlabel('Index')
        plt.ylabel('Value')
        
        # 현재 그래프를 PDF에 추가
        pdf.savefig()
        plt.close()  # 메모리 절약을 위해 그래프 닫기
        
        
data = np.load('/workspace/processed/Gyeongsan/test.npy', allow_pickle=True)
num_columns = data.shape[1]
with PdfPages('all_columns_visualization_test.pdf') as pdf:
    for i in range(num_columns):
        plt.figure()
        plt.plot(data[:, i])
        plt.title(f'Column {i+1} Visualization')
        plt.xlabel('Index')
        plt.ylabel('Value')
        
        # 현재 그래프를 PDF에 추가
        pdf.savefig()
        plt.close()  # 메모리 절약을 위해 그래프 닫기