import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from scipy import stats
import random  # 무작위 선택을 위한 모듈 추가

class DataProcessor:
    def __init__(self, train_data_dir='traindata', val_data_dir='valdata', output_dir='processed/Gyeongsan', outlier_factor=1.5, normalize=True):
        """
        데이터 처리를 위한 초기화 메서드.
        
        Args:
            train_data_dir (str): 트레인 데이터가 저장된 디렉토리 경로. 기본값은 'traindata'.
            val_data_dir (str): 검증 데이터가 저장된 디렉토리 경로. 기본값은 'valdata'.
            output_dir (str): 처리된 데이터를 저장할 디렉토리 경로. 기본값은 'processed/Gyeongsan'.
            outlier_factor (float): 이상치 감지를 위한 IQR 가중치. 기본값은 1.5.
            normalize (bool): 데이터 정규화 여부. 기본값은 True.
        """
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.output_dir = output_dir
        self.outlier_factor = outlier_factor  # 가중치 변수 추가
        self.normalize = normalize  # 정규화 옵션 추가
        self.setup_logging()  # 로깅 설정 메서드 호출
        os.makedirs(self.output_dir, exist_ok=True)  # 출력 디렉토리가 없으면 생성

    def setup_logging(self):
        """로깅 설정 메서드."""
        logging.basicConfig(
            level=logging.INFO,  # 로깅 레벨 설정 (INFO 이상 메시지 출력)
            format='%(asctime)s - %(levelname)s - %(message)s'  # 로그 메시지 포맷 설정
        )
        self.logger = logging.getLogger(__name__)  # 현재 모듈 이름으로 로거 생성

    def select_random_id(self, df):
        """
        데이터프레임에서 무작위로 하나의 id를 선택합니다.
        
        Args:
            df (pd.DataFrame): 데이터프레임.
        
        Returns:
            str: 선택된 id.
        """
        unique_ids = df['id'].unique()
        selected_id = random.choice(unique_ids)
        self.logger.info(f"무작위로 선택된 id: {selected_id}")
        return selected_id

    def read_csv_files(self, data_dir, selected_id=None):
        """
        지정된 디렉토리 내의 CSV 파일들을 읽어와 하나의 데이터프레임으로 결합.
        선택된 id가 있는 경우 해당 id의 데이터만 필터링.
        
        Args:
            data_dir (str): 데이터가 저장된 디렉토리 경로.
            selected_id (str, optional): 필터링할 id. 기본값은 None.
        
        Returns:
            pd.DataFrame: 결합된 데이터프레임.
        
        Raises:
            ValueError: 처리할 데이터가 없을 경우 발생.
        """
        all_data = []  # 모든 데이터프레임을 저장할 리스트
        columns = [
            'id',  # ID 컬럼
            'timestamp',  # 첫 번째 컬럼을 timestamp로 가정
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity'
        ]

        # 데이터 디렉토리 내의 모든 파일 탐색
        for file in os.listdir(data_dir):
            if not file.endswith('.csv'):
                continue  # CSV 파일이 아니면 건너뜀

            file_path = os.path.join(data_dir, file)
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

                # ID 및 timestamp 컬럼을 datetime 형식으로 변환
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # 데이터 정렬
                df = df.sort_values('timestamp')

                self.logger.info(f"파일 로드 완료: {file} (rows: {len(df)})")

                # 선택된 id가 있을 경우 필터링
                if selected_id is not None:
                    df = df[df['id'] == selected_id]
                    self.logger.info(f"선택된 id({selected_id})의 데이터 크기: {df.shape}")

                all_data.append(df)  # 데이터프레임을 리스트에 추가

            except Exception as e:
                self.logger.error(f"파일 처리 실패 {file}: {str(e)}")
                continue  # 에러 발생 시 해당 파일 건너뜀

        if not all_data:
            raise ValueError("처리할 데이터가 없습니다.")

        # ID별로 데이터를 그룹화하여 다양성을 확보 (선택된 id만 있을 경우 의미 없음)
        combined_df = pd.concat(all_data, ignore_index=True)

        # 중복 데이터 제거
        combined_df = combined_df.drop_duplicates()

        # timestamp 기준으로 최종 정렬
        combined_df = combined_df.sort_values(['id', 'timestamp'])

        self.logger.info(f"전체 데이터 크기: {combined_df.shape}")
        return combined_df

    def prepare_features(self, df):
        """
        시간 관련 특성을 생성하는 메서드.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임.
        
        Returns:
            pd.DataFrame: 특성이 추가된 데이터프레임.
        """
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        return df

    def sample_train_data(self, df):
        """
        동일한 ID에서 일정 간격으로 데이터를 샘플링하여 최대한 다양한 ID를 확보합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임.
        
        Returns:
            pd.DataFrame: 샘플링된 데이터프레임.
        """
        # ID를 그룹으로 설정하고 timestamp로 리샘플링
        df = df.set_index('timestamp')
        # 10분 간격으로 데이터 그룹화 (group_keys=False 추가)
        grouped = df.groupby(['id', pd.Grouper(freq='min')], group_keys=False)
        sampled_df = grouped.apply(lambda x: x.sample(n=1) if not x.empty else x)
        sampled_df = sampled_df.dropna().reset_index()  # 'timestamp' 컬럼이 한 번만 포함되도록 설정
        self.logger.info(f"샘플링된 트레인 데이터 크기: {sampled_df.shape}")
        return sampled_df

    def remove_outliers(self, df):
        """
        라벨을 계산하여 이상치로 판단된 데이터를 제거합니다.
        
        Args:
            df (pd.DataFrame): 샘플링된 데이터프레임.
        
        Returns:
            pd.DataFrame: 이상치가 제거된 데이터프레임.
        """
        # 라이블 생성 (슬로프, 크랙)
        slope_labels, crack_labels = self.generate_labels(df, outrate=1.2)
        # 슬로프와 크랙 레이블을 합쳐 이상치 여부 판단
        overall_outliers = np.any(slope_labels, axis=1) | np.any(crack_labels, axis=1)
        df['is_outlier'] = overall_outliers.astype(int)
        
        # 이상치인 데이터 제거
        clean_df = df[df['is_outlier'] == 0].drop(columns=['is_outlier'])
        self.logger.info(f"이상치 제거 후 트레인 데이터 ���기: {clean_df.shape}")
        return clean_df

    def normalize3(self, data, min_a=None, max_a=None):
        """
        데이터를 정규화하는 함수.

        Args:
            data (np.ndarray): 정규화할 데이터.
            min_a (np.ndarray, optional): 각 특성의 최소값. 기본값은 None.
            max_a (np.ndarray, optional): 각 특성의 최대값. 기본값은 None.

        Returns:
            tuple: 정규화된 데이터, min 배열, max 배열.
        """
        if min_a is None or max_a is None:
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
        else:
            data_min = min_a
            data_max = max_a

        # 정규화 방정식: (data - min) / (max - min)
        denominator = data_max - data_min

        # 분모가 0인 경우를 처리하여 정규화
        zero_denominator_mask = denominator == 0
        if np.any(zero_denominator_mask):
            self.logger.warning("일부 특성의 분모가 0이므로 랜덤 값으로 대체합니다.")
            denominator_safe = denominator.copy()
            denominator_safe[zero_denominator_mask] = 1  # 임시로 1로 설정하여 나눗셈을 가능하게 함
            normalized_data = (data - data_min) / denominator_safe
            # 분모가 0이었던 특성에 대해서는 랜덤 값으로 대체
            normalized_data[:, zero_denominator_mask] = np.random.uniform(0, 0.005, size=(data.shape[0], np.sum(zero_denominator_mask)))
        else:
            normalized_data = (data - data_min) / denominator

        return normalized_data, data_min, data_max

    def prepare_data_for_model(self, df, is_train=True, data_min=None, data_max=None):
        """
        모델 학습을 위한 데이터 준비 메서드.
        
        Args:
            df (pd.DataFrame): 결합된 데이터프레임.
            is_train (bool): 트레인 데이터인지 여부. True면 정규화 및 min/max 계산.
            data_min (np.ndarray, optional): 트레인 데이터의 최소값.
            data_max (np.ndarray, optional): 트레인 데이터의 최대값.
        
        Returns:
            tuple: 정규화된 슬로프 데이터, 크랙 데이터, min 배열, max 배열, 레이블 (is_train=False일 경우 None).
        """
        # 수정된 특성 리스트 (id 제외)
        features_all = [
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity',
            'hour', 'day_of_week', 'month'
        ]

        # 기울기 센서 특성 (id 제외)
        features_slope = [
            'initDegreeX', 'initDegreeY', 'degreeXAmount', 'degreeYAmount', 'temperature', 'humidity', 'hour', 'day_of_week', 'month'
        ]

        # 크랙 센서 특성 (id 제외)
        features_crack = [
            'initCrack', 'crackAmount', 'temperature', 'humidity', 'hour', 'day_of_week', 'month'
        ]

        # 데이터를 한 시간에 한 개씩 추출 (샘플링 유지)
        df.set_index('timestamp', inplace=True)
        hourly_df = df.resample('h').first().dropna().reset_index()

        if self.normalize:
            if is_train:
                # 기울기 센서 데이터 정규화
                normalized_slope, data_min_slope, data_max_slope = self.normalize3(hourly_df[features_slope].values)
                np.save(os.path.join(self.output_dir, 'train_slope.npy'), normalized_slope)
                np.save(os.path.join(self.output_dir, 'min_slope.npy'), data_min_slope)
                np.save(os.path.join(self.output_dir, 'max_slope.npy'), data_max_slope)
                self.logger.info(f"트레인 슬로프 데이터 크기: {normalized_slope.shape}")

                # 크랙 센서 데이터 정규화
                normalized_crack, data_min_crack, data_max_crack = self.normalize3(hourly_df[features_crack].values)
                np.save(os.path.join(self.output_dir, 'train_crack.npy'), normalized_crack)
                np.save(os.path.join(self.output_dir, 'min_crack.npy'), data_min_crack)
                np.save(os.path.join(self.output_dir, 'max_crack.npy'), data_max_crack)
                self.logger.info(f"트레인 크랙 데이터 크기: {normalized_crack.shape}")
                                    
                return (normalized_slope, data_min_slope, data_max_slope), (normalized_crack, data_min_crack, data_max_crack)
            else:
                if data_min is None or data_max is None:
                    raise ValueError("트레인 데이터의 min과 max 값이 필요합니다.")
                
                # 기울기 센서 데이터 정규화 (트레인 데이터의 min과 max 사용)
                normalized_slope, _, _ = self.normalize3(hourly_df[features_slope].values, min_a=data_min[:9], max_a=data_max[:9])
                np.save(os.path.join(self.output_dir, 'test_slope.npy'), normalized_slope)

                # 크랙 센서 데이터 정규화 (트레인 데이터의 min과 max 사용)
                normalized_crack, _, _ = self.normalize3(hourly_df[features_crack].values, min_a=data_min[9:], max_a=data_max[9:])
                np.save(os.path.join(self.output_dir, 'test_crack.npy'), normalized_crack)

                # 레이블 생성 및 분리
                slope_labels, crack_labels = self.generate_labels(hourly_df)
                np.save(os.path.join(self.output_dir, 'labels_test_slope.npy'), slope_labels)
                np.save(os.path.join(self.output_dir, 'labels_test_crack.npy'), crack_labels)

                self.logger.info(f"테스트 슬로프 데이터 크기: {normalized_slope.shape}")
                self.logger.info(f"테스트 크랙 데이터 크기: {normalized_crack.shape}")
                self.logger.info(f"레벨 슬로프 데이터 크기: {slope_labels.shape}")
                self.logger.info(f"레벨 크랙 데이터 크기: {crack_labels.shape}")

                return normalized_slope, normalized_crack, slope_labels, crack_labels
        else:
            # 정규화 없이 데이터 저장
            if is_train:
                data = hourly_df[features_all].values
                np.save(os.path.join(self.output_dir, 'train.npy'), data)
                self.logger.info(f"트레인 데이터 (정규화 없음) 크기: {data.shape}")
                return data, None, None
            else:
                data = hourly_df[features_all].values
                labels = self.generate_labels(hourly_df)
                np.save(os.path.join(self.output_dir, 'test.npy'), data)
                np.save(os.path.join(self.output_dir, 'test_slope_labels.npy'), labels[0])
                np.save(os.path.join(self.output_dir, 'test_crack_labels.npy'), labels[1])
                self.logger.info(f"테스트 데이터 (정규화 없음) 크기: {data.shape}")
                self.logger.info(f"레벨 슬로프 데이터 크기: {labels[0].shape}")
                self.logger.info(f"레벨 크랙 데이터 크기: {labels[1].shape}")
                return data, labels[0], labels[1]

    def generate_labels(self, df, outrate=None):
        """
        레이블 항목에 대한 레이블 데이터를 생성하는 함수.
        
        Args:
            df (pd.DataFrame): 데이터프레임.
        
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

            if outrate is None:
                # 이상치 기준 설정 (1.5 * IQR)
                lower_bound = Q1 - self.outlier_factor * IQR
                upper_bound = Q3 + self.outlier_factor * IQR
            else:
                # 이상치 기준 설정 (outrate * IQR)
                lower_bound = Q1 - outrate * IQR
                upper_bound = Q3 + outrate * IQR
            
            # 이상치 여부 판단 (0: 정상, 1: 이상치)
            anomaly = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).astype(int)

            # 레이블 데이터프레임에 추가
            label_df[f'anomaly_{feature}'] = anomaly

        # 슬로프 관련 레이블 추출
        slope_features = [
            'anomaly_initDegreeX', 'anomaly_initDegreeY',
            'anomaly_degreeXAmount', 'anomaly_degreeYAmount',
            'anomaly_temperature', 'anomaly_humidity',
            'anomaly_hour', 'anomaly_day_of_week', 'anomaly_month'
        ]
        slope_labels = label_df[slope_features].values

        # 크랙 관련 레이블 추출
        crack_features = [
            'anomaly_initCrack', 'anomaly_crackAmount',
            'anomaly_temperature', 'anomaly_humidity',
            'anomaly_hour', 'anomaly_day_of_week', 'anomaly_month'
        ]
        crack_labels = label_df[crack_features].values

        return slope_labels, crack_labels  # 슬로프와 크랙 레이블을 별도로 반환

    def process_train(self, selected_id=None):
        """
        트레인 데이터를 처리하는 메서드.
        
        Args:
            selected_id (str, optional): 처리할 특정 id. 기본값은 None.
        
        Returns:
            tuple: 정규화된 트레인 슬로프 데이터, 트레인 크랙 데이터.
        
        Raises:
            Exception: 데이터 처리 중 발생한 예외.
        """
        try:
            self.logger.info("트레인 데이터 처리 시작")
            df = self.read_csv_files(self.train_data_dir, selected_id=selected_id)  # 트레인 CSV 파일 읽기 및 결합

            if selected_id is None:
                # 선택된 id가 없는 경우 무작위로 id 선택
                selected_id = self.select_random_id(df)
                df = df[df['id'] == selected_id]
                self.logger.info(f"무작위로 선택된 id({selected_id})의 데이터 사용")

            # 시간 관련 특성 생성
            df = self.prepare_features(df)

            # 이상치 제거
            clean_df = self.remove_outliers(df)

            # 데이터 샘플링
            sampled_df = self.sample_train_data(clean_df)

            # 데이터 전처리 및 정규화
            train_slope, train_crack = self.prepare_data_for_model(sampled_df, is_train=True)
            self.logger.info("트레인 데이터 처리 완료")
            return train_slope, train_crack
        except Exception as e:
            self.logger.error(f"트레인 데이터 처리 실패: {str(e)}")
            raise  # 예외를 상위로 전달

    def process_val(self):
        """
        검증 데이터를 처리하는 메서드.
        
        Returns:
            tuple: 정규화된 검증 슬로프 데이터, 검증 크랙 데이터, 레이블 배열.
        
        Raises:
            Exception: 데이터 처리 중 발생한 예외.
        """
        try:
            self.logger.info("검증 데이터 처리 시작")
            df = self.read_csv_files(self.val_data_dir)  # 검증 CSV 파일 읽기 및 결합

            # 시간 관련 특성 생성
            df = self.prepare_features(df)
            
            # 데이터를 한 시간에 한 개씩 추출 (샘플링 유지)
            df.set_index('timestamp', inplace=True)
            hourly_df = df.resample('h').first().dropna().reset_index()

            # 트레인 데이터의 min과 max 로드
            data_min_slope = np.load(os.path.join(self.output_dir, 'min_slope.npy'))
            data_max_slope = np.load(os.path.join(self.output_dir, 'max_slope.npy'))
            data_min_crack = np.load(os.path.join(self.output_dir, 'min_crack.npy'))
            data_max_crack = np.load(os.path.join(self.output_dir, 'max_crack.npy'))
            
            # 슬로프 및 크랙 데이터 정규화
            normalized_slope, normalized_crack, slope_labels, crack_labels = self.prepare_data_for_model(hourly_df, is_train=False, data_min=np.concatenate((data_min_slope, data_min_crack)), data_max=np.concatenate((data_max_slope, data_max_crack)))
            
            self.logger.info("검증 데이터 처리 완료")
            return normalized_slope, normalized_crack, slope_labels, crack_labels
        except Exception as e:
            self.logger.error(f"검증 데이터 처리 실패: {str(e)}")
            raise  # 예외를 상위로 전달

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="데이터 프로세서 실행 스크립트")
    parser.add_argument('--outlier_factor', type=float, default=2.5, help='이상치 감지를 위한 IQR 가중치')
    parser.add_argument('--normalize', action='store_true', help='데이터 정규화 여부')
    parser.add_argument('--train_id', type=str, default=None, help='트레인 데이터 처리 시 사용할 특정 id')
    parser.add_argument('--val_id', type=str, default=None, help='검증 데이터 처리 시 사용할 특정 id')

    args = parser.parse_args()
    
    # 인스턴스 생성 시 normalize 옵션 적용
    args.normalize = True
    
    processor_custom = DataProcessor(outlier_factor=args.outlier_factor, normalize=args.normalize)
    # train_data = processor_custom.process_train(selected_id=args.train_id)
    val_data = processor_custom.process_val()
    
    