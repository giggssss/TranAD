import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from scipy import stats

class DataProcessor:
    def __init__(self, data_dir='data_sample', output_dir='processed/Gyeongsan'):
        """
        데이터 처리를 위한 초기화
        Args:
            data_dir: 원본 데이터가 있는 디렉토리
            output_dir: 처리된 데이터를 저장할 디렉토리
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_logging()
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def read_csv_files(self):
        """CSV 파일들을 읽어서 하나의 데이터프레임으로 결합"""
        all_data = []
        columns = [
            'timestamp',  # 첫 번째 컬럼을 timestamp로 가정
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity'
        ]
        
        for file in os.listdir(self.data_dir):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(self.data_dir, file)
            try:
                # CSV 파일 읽기
                df = pd.read_csv(
                    file_path, 
                    sep='|',  # 구분자
                    names=columns,  # 컬럼명 지정
                    skiprows=1,  # 헤더 제외
                    encoding='utf-8',  # 인코딩
                    parse_dates=['timestamp']  # timestamp 파싱
                )
                
                # timestamp 처리
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 데이터 정렬
                df = df.sort_values('timestamp')
                
                self.logger.info(f"파일 로드 완료: {file} (rows: {len(df)})")
                all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"파일 처리 실패 {file}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("처리할 데이터가 없습니다.")
            
        # 데이터 결합
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 중복 제거
        combined_df = combined_df.drop_duplicates()
        
        # 시간순 정렬
        combined_df = combined_df.sort_values('timestamp')
        
        self.logger.info(f"전체 데이터 크기: {combined_df.shape}")
        return combined_df
    
    def prepare_data_for_model(self, df):
        """모델 학습을 위한 데이터 준비"""
        # 타임스탬프를 특성으로 변환
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # 최빈값 계산을 위해 월별 그룹화
        df['month_period'] = df['timestamp'].dt.to_period('M')
        
        # 각 월의 최빈값 계산
        def safe_mode(x):
            if len(x) == 0:
                return np.nan
            mode_result = stats.mode(x, nan_policy='omit')
            if mode_result.count > 0:
                return mode_result.mode
            else:
                return np.nan
        
        mode_displacements = df.groupby('month_period')[['initDegreeX', 'initDegreeY', 'initDegreeZ']].agg(safe_mode)
        
        # 월별로 최빈값을 매핑
        df = df.join(mode_displacements, on='month_period', rsuffix='_mode')
        
        # 기울기 계산을 위해 X, Y, Z의 이동 평균 계산 (3개월 누적)
        def calculate_slope(x):
            try:
                if len(x) < 2:
                    return np.nan
                return np.polyfit(range(len(x)), x, 1)[0]
            except np.linalg.LinAlgError:
                return np.nan
        
        df['slope_X'] = df['initDegreeX'].rolling(window=90, min_periods=2).apply(calculate_slope, raw=True)
        df['slope_Y'] = df['initDegreeY'].rolling(window=90, min_periods=2).apply(calculate_slope, raw=True)
        df['slope_Z'] = df['initDegreeZ'].rolling(window=90, min_periods=2).apply(calculate_slope, raw=True)
        
        # 사분위수 계산 (IQR)
        Q1_X, Q3_X = df['slope_X'].quantile([0.25, 0.75])
        IQR_X = Q3_X - Q1_X
        Q1_Y, Q3_Y = df['slope_Y'].quantile([0.25, 0.75])
        IQR_Y = Q3_Y - Q1_Y
        Q1_Z, Q3_Z = df['slope_Z'].quantile([0.25, 0.75])
        IQR_Z = Q3_Z - Q1_Z
        
        # 이상치 기준 설정
        lower_X, upper_X = Q1_X - 1.5 * IQR_X, Q3_X + 1.5 * IQR_X
        lower_Y, upper_Y = Q1_Y - 1.5 * IQR_Y, Q3_Y + 1.5 * IQR_Y
        lower_Z, upper_Z = Q1_Z - 1.5 * IQR_Z, Q3_Z + 1.5 * IQR_Z
        
        # 이상치 필터링
        df['outlier_X'] = ((df['slope_X'] < lower_X) | (df['slope_X'] > upper_X)).astype(int)
        df['outlier_Y'] = ((df['slope_Y'] < lower_Y) | (df['slope_Y'] > upper_Y)).astype(int)
        df['outlier_Z'] = ((df['slope_Z'] < lower_Z) | (df['slope_Z'] > upper_Z)).astype(int)
        
        # |계측값 - 최빈값| 계산
        df['diff_X'] = np.abs(df['initDegreeX'] - df['initDegreeX_mode'])
        df['diff_Y'] = np.abs(df['initDegreeY'] - df['initDegreeY_mode'])
        df['diff_Z'] = np.abs(df['initDegreeZ'] - df['initDegreeZ_mode'])
        
        # F(Q1), F(Q3) 정의 (예: Q1과 Q3 그대로 사용)
        F_Q1_X, F_Q3_X = Q1_X, Q3_X
        F_Q1_Y, F_Q3_Y = Q1_Y, Q3_Y
        F_Q1_Z, F_Q3_Z = Q1_Z, Q3_Z
        
        # 이상치 정의
        df['anomaly_X'] = ((df['diff_X'] < F_Q1_X) | (df['diff_X'] > F_Q3_X)).astype(int)
        df['anomaly_Y'] = ((df['diff_Y'] < F_Q1_Y) | (df['diff_Y'] > F_Q3_Y)).astype(int)
        df['anomaly_Z'] = ((df['diff_Z'] < F_Q1_Z) | (df['diff_Z'] > F_Q3_Z)).astype(int)
        
        # 전체 레이블 결합
        df['anomaly'] = (df['outlier_X'] | df['outlier_Y'] | df['outlier_Z'] | 
                         df['anomaly_X'] | df['anomaly_Y'] | df['anomaly_Z']).astype(int)
        
        # 초 단위로 데이터를 순차적으로 train과 test로 분할
        train_df = df.iloc[::2].reset_index(drop=True)
        test_df = df.iloc[1::2].reset_index(drop=True)
        
        # 특성 선택 (타임스탬프 관련 특성 포함)
        features = [
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity',
            'hour', 'day_of_week', 'month'
        ]
        
        # 데이터 정규화
        train_data = train_df[features].values
        test_data = test_df[features].values
        
        # 정규화 파라미터 계산 (train 데이터 기준)
        data_mean = np.mean(train_data, axis=0)
        data_std = np.std(train_data, axis=0)
        normalized_train = (train_data - data_mean) / (data_std + 1e-10)
        normalized_test = (test_data - data_mean) / (data_std + 1e-10)
        
        # 레이블 추출 (test 데이터에 대해서만)
        labels = test_df['anomaly'].values.reshape(-1, 1)
        # 레이블의 shape을 (n, x)로 변경
        labels = np.tile(labels, (1, normalized_test.shape[1]))
        
        # train과 test의 샘플 수가 동일하게 맞추기
        min_samples = min(normalized_train.shape[0], normalized_test.shape[0])
        normalized_train = normalized_train[:min_samples]
        normalized_test = normalized_test[:min_samples]
        labels = labels[:min_samples]
        
        # 데이터 저장
        np.save(os.path.join(self.output_dir, 'train.npy'), normalized_train)
        np.save(os.path.join(self.output_dir, 'test.npy'), normalized_test)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels)
        np.save(os.path.join(self.output_dir, 'mean.npy'), data_mean)
        np.save(os.path.join(self.output_dir, 'std.npy'), data_std)
        
        self.logger.info(f"학습 데이터 크기: {normalized_train.shape}")
        self.logger.info(f"테스트 데이터 크기: {normalized_test.shape}")
        self.logger.info(f"레이블 데이터 크기: {labels.shape}")
        
        return normalized_train, normalized_test, labels
    
    def process(self):
        """전체 데이터 처리 프로세스 실행"""
        try:
            self.logger.info("데이터 처리 시작")
            df = self.read_csv_files()
            result = self.prepare_data_for_model(df)
            self.logger.info("데이터 처리 완료")
            return result
        except Exception as e:
            self.logger.error(f"데이터 처리 실패: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor()
    train_data, test_data, labels = processor.process()