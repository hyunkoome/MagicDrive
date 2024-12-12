import os


def make_dir(folder_path):
    try:
        # 부모 폴더까지 모두 생성
        os.makedirs(folder_path, exist_ok=True)  # 존재해도 에러 없이 통과
        # print(f"폴더 생성 완료 또는 이미 존재: {folder_path}")
    except Exception as e:
        print(f"폴더 생성 중 오류 발생: {e}")