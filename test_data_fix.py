
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_pipeline.parser import StmechDataParser

def test_data_loading():
    print("--- 데이터 로딩 경로 테스트 ---")
    parser = StmechDataParser()
    print(f"기본 설정 경로: {parser.filepath}")
    
    try:
        X, y, feats = parser.load_and_preprocess()
        print(f"성공! 데이터 로드 완료.")
        print(f"행(Row) 수: {len(X)}")
        print(f"열(Column) 수: {len(feats)}")
        
        if len(X) == 100:
            print("경고: 여전히 100줄만 읽힙니다. (가짜 데이터일 가능성 있음)")
        else:
            print(f"확인: 실제 데이터 {len(X)}줄을 정상적으로 읽어왔습니다.")
            
    except Exception as e:
        print(f"실패: 데이터를 로드할 수 없습니다. 에러: {e}")

if __name__ == "__main__":
    test_data_loading()
