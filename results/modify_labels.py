import numpy as np

def modify_labels():
    # labels.npy 파일 로드
    labels = np.load('processed/Gyeongsan/labels.npy')
    
    # 2번과 6번 차원의 인덱스는 1과 5 (0부터 시작)
    labels[:, 2] = 0
    labels[:, 6] = 0
    
    # 수정된 배열 저장
    np.save('processed/Gyeongsan/labels.npy', labels)

if __name__ == "__main__":
    modify_labels() 