import json
import glob
import os

# 바꿀 jsonl 파일 리스트
jsonl_files = [ "train_2.jsonl"]#"validation_2.jsonl",

def convert_ext_to_mtx(path):
    # .mid, .MID 등 모두 .mtx로
    base, ext = os.path.splitext(path)
    return base + ".mtf"

for jsonl_path in jsonl_files:
    # 안전하게 백업본 저장
    os.rename(jsonl_path, jsonl_path + ".bak")
    print(jsonl_path)
    with open(jsonl_path + ".bak", "r", encoding="utf-8") as r, \
         open(jsonl_path, "w", encoding="utf-8") as w:
        for line in r:
            item = json.loads(line)
            # filepaths가 list 또는 str일 수 있음
            if "filepaths" in item:
                if isinstance(item["filepaths"], list):
                    item["filepaths"] = [convert_ext_to_mtx(fp) for fp in item["filepaths"]]
                elif isinstance(item["filepaths"], str):
                    item["filepaths"] = convert_ext_to_mtx(item["filepaths"])
            w.write(json.dumps(item, ensure_ascii=False) + "\n")

print("모든 파일의 filepaths 확장자를 .mtx로 변환 완료!")
