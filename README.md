# maap.github.io
MAAP LAB Github Site

# 🎵 Jamendo Music Crawler

Jamendo API를 활용해 음악 메타데이터와 오디오 파일을 수집하고, CSV로 저장하거나 `.wav`로 다운로드하는 파이프라인입니다.  
장르(genre), 악기(instrument), 분위기(mood/theme) 태그별로 대규모 음악 데이터를 효율적으로 수집할 수 있습니다.

---

## 📦 주요 기능

- ✅ Jamendo API 기반 음악 검색
- ✅ 장르, 분위기, 악기별 자동 반복 수집
- ✅ 메타데이터를 CSV로 저장
- ✅ 오디오 파일 `.wav`로 다운로드 (선택)
- ✅ ThreadPool을 활용한 병렬 다운로드
- ✅ 클린 코드 스타일, 테스트 및 확장 용이

---

## 🏗️ 프로젝트 구조

```
jamendo_project/
│
├── data
    ├── AudioCrawling.py # 오디오 데이터 수집
    └── dataset_hf.py # 데이터 로드
├── tests
    └──test_dataloader # 데이터로더 테스트
├── scripts
    ├── dataloder.sh
    └── downloader.sh
├── requirements.txt
└── README.md
```

---

## ⚙️ 설치 및 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

필요한 주요 패키지:
- `requests`
- `argparse-dataclass`
- `tqdm`

### 2. 실행 방법

```bash
python main.py --download False --save_path ./downloads --step 200
```

### 주요 옵션 (전부 `JamendoMusicConfig`에서 설정됨)

| 옵션명            | 타입    | 설명 |
|-------------------|---------|------|
| `--client_id`      | str     | Jamendo API 클라이언트 ID (**필수**) |
| `--download`       | bool    | 실제 오디오 파일 다운로드 여부 (`True` 시 다운로드) |
| `--save_path`      | str     | 다운로드 및 CSV 저장 경로 |
| `--offset_start`   | int     | 검색 시작 오프셋 (기본 0) |
| `--offset_end`     | int     | 검색 종료 오프셋 (기본 1,000,000) |
| `--step`           | int     | 한 번에 요청할 트랙 수 (최대 200) |
| `--num_threads`    | int     | 병렬 다운로드 시 사용할 쓰레드 수 |
| `--tags`           | Dict    | 태그 그룹 정의 (`genre`, `instrument`, `mood/theme`) |

---

## 🛠 사용 예시

### 메타데이터만 저장 (다운로드 X)

```bash
python main.py --download False
```

### 오디오 파일도 다운로드

```bash
python main.py --download True --step 100
```

---

## 📄 CSV 예시 형식

`downloads/metadata.csv`에 다음과 같은 형식으로 저장됩니다:

| name      | duration | artist_name | waveform | tag     | audio_path                  |
|-----------|----------|-------------|----------|---------|-----------------------------|
| track123  | 180.0    | DJ Jam      | ...      | rock    | downloads/rock_track123.wav |

---

## 📌 참고사항

- `.wav` 저장 시 파일 이름은 특수 문자 제거 후 `장르_트랙이름.wav` 형태로 저장됩니다.
- 동일 장르에 중복된 트랙이 많을 수 있으므로, 중복 체크나 `name + artist_name` 기준 중복 제거 추천.
- `audiodownload_allowed` 값이 False인 경우 다운로드/저장은 건너뜁니다.

---

## 🧪 테스트

테스트 코드도 별도로 작성할 수 있으며, 특히 `cleaner.py`, `api.py` 등에 대한 단위 테스트 작성이 용이합니다.

---

## 🤝 기여

- 태그 필터링 전략 개선
- 라이센스 필터링 (CC BY 등)
- 유사곡 제거 / 중복 제거 로직
- 미디어 길이 기준 샘플링 개선

모두 환영합니다 🙌

---

## 📜 라이센스

> 본 코드는 Jamendo의 [Open API](https://developer.jamendo.com/v3.0) 사용 정책을 따릅니다. 수집된 음원은 상업적 용도로 사용할 수 없습니다.
