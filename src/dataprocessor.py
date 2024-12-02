import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from scipy import stats

class DataProcessor:
    def __init__(self, data_dir='data_sample', output_dir='processed/Gyeongsan'):
        """
        데이터 처리를 위한 초기화 메서드.
        
        Args:
            data_dir (str): 원본 데이터가 저장된 디렉토리 경로. 기본값은 'data_sample'.
            output_dir (str): 처리된 데이터를 저장할 디렉토리 경로. 기본값은 'processed/Gyeongsan'.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.setup_logging()  # 로깅 설정 메서드 호출
        os.makedirs(self.output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성

    def setup_logging(self):
        """로깅 설정 메서드."""
        logging.basicConfig(
            level=logging.INFO,  # 로깅 레벨 설정 (INFO 이상 메시지 출력)
            format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 포맷 설정
        )
        self.logger = logging.getLogger(__name__)  # 현재 모듈 이름으로 로거 생성

    def read_csv_files(self):
        """
        지정된 디렉토리 내의 CSV 파일들을 읽어와 하나의 데이터프레임으로 결합.
        
        Returns:
            pd.DataFrame: 결합된 데이터프레임.
        
        Raises:
            ValueError: 처리할 데이터가 없을 경우 발생.
        """
        all_data = []  # 모든 데이터프레임을 저장할 리스트
        columns = [
            'timestamp',  # 첫 번째 컬럼을 timestamp로 가정
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity'
        ]

        # 데이터 디렉토리 내의 모든 파일 탐색
        for file in os.listdir(self.data_dir):
            if not file.endswith('.csv'):
                continue  # CSV 파일이 아니면 건너뜀

            file_path = os.path.join(self.data_dir, file)
            try:
                # CSV 파일 읽기
                df = pd.read_csv(
                    file_path, 
                    sep='|',  # 구분자 설정
                    names=columns,  # 컬럼명 지정
                    skiprows=1,  # 헤더 제외
                    encoding='utf-8',  # 인코딩 설정
                    parse_dates=['timestamp']  # timestamp 컬럼을 날짜 형식으로 파싱
                )

                # timestamp 컬럼을 datetime 형식으로 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # 데이터 정렬
                df = df.sort_values('timestamp')

                self.logger.info(f"파일 로드 완료: {file} (rows: {len(df)})")
                all_data.append(df)  # 데이터프레임을 리스트에 추가

            except Exception as e:
                self.logger.error(f"파일 처리 실패 {file}: {str(e)}")
                continue  # 에러 발생 시 해당 파일 건너뜀

        if not all_data:
            raise ValueError("처리할 데이터가 없습니다.")

        # 모든 데이터프레임을 하나로 결합
        combined_df = pd.concat(all_data, ignore_index=True)

        # 중복 데이터 제거
        combined_df = combined_df.drop_duplicates()

        # timestamp 기준으로 최종 정렬
        combined_df = combined_df.sort_values('timestamp')

        self.logger.info(f"전체 데이터 크기: {combined_df.shape}")
        return combined_df

    def prepare_data_for_model(self, df):
        """
        모델 학습을 위한 데이터 준비 메서드.
        
        Args:
            df (pd.DataFrame): 결합된 데이터프레임.
        
        Returns:
            tuple: 정규화된 학습 데이터, 테스트 데이터, 레이블 배열.
        """
        # 타임스탬프를 기반으로 다양한 시간 관련 특성 생성
        df['hour'] = df['timestamp'].dt.hour  # 시간 추출
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 요일 추출
        df['month'] = df['timestamp'].dt.month  # 월 추출

        # 최빈값 계산을 위해 월별로 그룹화
        df['month_period'] = df['timestamp'].dt.to_period('M')  # 월 단위 기간 생성

        # 각 월별로 initDegreeX, initDegreeY, initDegreeZ의 최빈값 계산
        def safe_mode(x):
            """
            최빈값을 안전하게 계산하는 함수.
            
            Args:
                x (pd.Series): 계산할 데이터 시리즈.
            
            Returns:
                float: 최빈값 또는 NaN.
            """
            modes = x.mode()
            if modes.empty:
                return np.nan
            return modes.iloc[0]

        # 월별 최빈값 계산
        mode_displacements = df.groupby('month_period')[['initDegreeX', 'initDegreeY', 'initDegreeZ']].agg(safe_mode)

        # 월별 최빈값을 원본 데이터프레임에 매핑
        df = df.join(mode_displacements, on='month_period', rsuffix='_mode')

        # 기울기 계산을 위한 함수 정의 (효율적인 기울기 계산)
        def compute_slope(series, window):
            """
            주어진 시계열 데이터의 기울기를 효율적으로 계산하는 함수.
            
            Args:
                series (pd.Series): 시계열 데이터.
                window (int): 윈도우 크기.
            
            Returns:
                pd.Series: 기울기 값.
            """
            # x값은 0,1,2,...,window-1
            x = np.arange(window)
            sum_x = x.sum()
            sum_x2 = (x ** 2).sum()
            denominator = window * sum_x2 - sum_x ** 2

            # Rolling 합계 계산
            sum_y = series.rolling(window).sum()
            sum_xy = series.rolling(window).apply(lambda y: np.dot(x, y), raw=True)

            # 기울기 계산
            slope = (window * sum_xy - sum_x * sum_y) / denominator
            return slope

        window_size = 90  # 이동창 크기 설정

        # X, Y, Z 축에 대한 기울기 계산 (90일 이동창)
        df['slope_X'] = compute_slope(df['initDegreeX'], window_size)
        df['slope_Y'] = compute_slope(df['initDegreeY'], window_size)
        df['slope_Z'] = compute_slope(df['initDegreeZ'], window_size)

        # 사분위수 계산 (IQR)
        Q1_X, Q3_X = df['slope_X'].quantile([0.25, 0.75])
        IQR_X = Q3_X - Q1_X
        Q1_Y, Q3_Y = df['slope_Y'].quantile([0.25, 0.75])
        IQR_Y = Q3_Y - Q1_Y
        Q1_Z, Q3_Z = df['slope_Z'].quantile([0.25, 0.75])
        IQR_Z = Q3_Z - Q1_Z

        # 이상치 기준 설정 (IQR 방식)
        lower_X, upper_X = Q1_X - 1.5 * IQR_X, Q3_X + 1.5 * IQR_X
        lower_Y, upper_Y = Q1_Y - 1.5 * IQR_Y, Q3_Y + 1.5 * IQR_Y
        lower_Z, upper_Z = Q1_Z - 1.5 * IQR_Z, Q3_Z + 1.5 * IQR_Z

        # 이상치 여부를 나타내는 컬럼 생성 (0 또는 1)
        df['outlier_X'] = ((df['slope_X'] < lower_X) | (df['slope_X'] > upper_X)).astype(int)
        df['outlier_Y'] = ((df['slope_Y'] < lower_Y) | (df['slope_Y'] > upper_Y)).astype(int)
        df['outlier_Z'] = ((df['slope_Z'] < lower_Z) | (df['slope_Z'] > upper_Z)).astype(int)

        # 계측값과 최빈값의 차이 계산
        df['diff_X'] = np.abs(df['initDegreeX'] - df['initDegreeX_mode'])
        df['diff_Y'] = np.abs(df['initDegreeY'] - df['initDegreeY_mode'])
        df['diff_Z'] = np.abs(df['initDegreeZ'] - df['initDegreeZ_mode'])

        # 사분위수 값 정의 (기존 Q1, Q3 사용)
        F_Q1_X, F_Q3_X = Q1_X, Q3_X
        F_Q1_Y, F_Q3_Y = Q1_Y, Q3_Y
        F_Q1_Z, F_Q3_Z = Q1_Z, Q3_Z

        # 이상치 정의 (차이가 Q1 미만 또는 Q3 초과인 경우)
        df['anomaly_X'] = ((df['diff_X'] < F_Q1_X) | (df['diff_X'] > F_Q3_X)).astype(int)
        df['anomaly_Y'] = ((df['diff_Y'] < F_Q1_Y) | (df['diff_Y'] > F_Q3_Y)).astype(int)
        df['anomaly_Z'] = ((df['diff_Z'] < F_Q1_Z) | (df['diff_Z'] > F_Q3_Z)).astype(int)

        # 특성 선택 (타임스탬프 관련 특성 포함)
        features = [
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity',
            'hour', 'day_of_week', 'month'
        ]

        # 데이터를 한 시간에 한 개씩 추출
        df.set_index('timestamp', inplace=True)
        hourly_df = df.resample('H').first().dropna().reset_index()

        # Train 데이터와 Test 데이터 분할
        train_df = hourly_df.iloc[::2].reset_index(drop=True)  # 짝수 인덱스 (Train)
        test_df = hourly_df.iloc[1::2].reset_index(drop=True)  # 홀수 인덱스 (Test)

        # 기존에 정규화된 데이터를 -1에서 1 사이로 다시 정규화
        train_data = train_df[features].values
        test_data = test_df[features].values

        # 기존 정규화된 데이터의 최소값과 최대값을 계산
        data_min = train_data.min(axis=0)
        data_max = train_data.max(axis=0)

        # -1과 1 사이로 정규화
        normalized_train = 2 * (train_data - data_min) / (data_max - data_min + 1e-10) - 1
        normalized_test = 2 * (test_data - data_min) / (data_max - data_min + 1e-10) - 1

        # Test 데이터에 대해서만 레이블 생성
        labels = self.generate_labels(test_df)

        # 정규화된 데이터와 레이블을 NumPy 파일로 저장
        np.save(os.path.join(self.output_dir, 'train.npy'), normalized_train)
        np.save(os.path.join(self.output_dir, 'test.npy'), normalized_test)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels)
        np.save(os.path.join(self.output_dir, 'min.npy'), data_min)
        np.save(os.path.join(self.output_dir, 'max.npy'), data_max)

        # 처리된 데이터의 크기 로그 출력
        self.logger.info(f"학습 데이터 크기: {normalized_train.shape}")
        self.logger.info(f"테스트 데이터 크기: {normalized_test.shape}")
        self.logger.info(f"레이블 데이터 크기: {labels.shape}")

        return normalized_train, normalized_test, labels

    def generate_labels(self, df):
        """
        레이블 항목에 대한 레이블 데이터를 생성하는 함수.
        
        Args:
            df (pd.DataFrame): 테스트 데이터프레임.
        
        Returns:
            np.ndarray: 생성된 레이블 배열. 각 항목마다 0 또는 1의 값을 가짐.
        """
        # 레이블을 저장할 빈 데이터프레임 생성
        label_df = pd.DataFrame()

        # 이상치를 감지할 모든 특성 목록
        features = [
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity',
            'hour', 'day_of_week', 'month'
        ]

        for feature in features:
            # 각 특성별 사분위수 계산
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1

            # 이상치 기준 설정 (1.5 * IQR)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 이상치 여부 판단 (0: 정상, 1: 이상치)
            anomaly = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).astype(int)

            # 레이블 데이터프레임에 추가
            label_df[f'anomaly_{feature}'] = anomaly

        # 레이블을 NumPy 배열로 변환
        labels = label_df.values  # Shape: (샘플 수, 13)

        return labels

    def process(self):
        """
        전체 데이터 처리 프로세스를 실행하는 메서드.
        
        Returns:
            tuple: 정규화된 학습 데이터, 테스트 데이터, 레이블 배열.
        
        Raises:
            Exception: 데이터 처리 중 발생한 예외.
        """
        try:
            self.logger.info("데이터 처리 시작")
            df = self.read_csv_files()  # CSV 파일 읽기 및 결합
            result = self.prepare_data_for_model(df)  # 데이터 전처리 및 정규화
            self.logger.info("데이터 처리 완료")
            return result
        except Exception as e:
            self.logger.error(f"데이터 처리 실패: {str(e)}")
            raise  # 예외를 상위로 전달

if __name__ == "__main__":
    processor = DataProcessor()  # DataProcessor 인스턴스 생성
    train_data, test_data, labels = processor.process()  # 데이터 처리 실행