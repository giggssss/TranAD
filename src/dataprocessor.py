import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

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
            'id', 'timestamp', 
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
                    encoding='utf-8'  # 인코딩
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
        # 학습에 사용할 특성
        features = [
            'initDegreeX', 'initDegreeY', 'initDegreeZ',
            'degreeXAmount', 'degreeYAmount', 'degreeZAmount',
            'initCrack', 'crackAmount', 'temperature', 'humidity'
        ]
        
        # 특성 데이터 추출
        model_input = df[features].values
        
        # 데이터 정규화
        data_mean = np.mean(model_input, axis=0)
        data_std = np.std(model_input, axis=0)
        normalized_data = (model_input - data_mean) / (data_std + 1e-10)
        
        # 학습/테스트 데이터 분할 (80:20)
        train_size = int(len(normalized_data) * 0.8)
        train_data = normalized_data[:train_size]
        test_data = normalized_data[train_size:]
        
        # 레이블 생성
        labels = np.zeros((len(test_data), len(features)))
        
        # 데이터 저장
        np.save(os.path.join(self.output_dir, 'train.npy'), train_data)
        np.save(os.path.join(self.output_dir, 'test.npy'), test_data)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels)
        
        # 정규화 파라미터 저장
        np.save(os.path.join(self.output_dir, 'mean.npy'), data_mean)
        np.save(os.path.join(self.output_dir, 'std.npy'), data_std)
        
        self.logger.info(f"학습 데이터 크기: {train_data.shape}")
        self.logger.info(f"테스트 데이터 크기: {test_data.shape}")
        
        return train_data, test_data, labels
    
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