# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda가 설치되어 있지 않습니다."
    echo "[INFO] https://docs.conda.io/en/latest/miniconda.html 에서 설치하세요."
    exit 1
fi


# Conda 환셩 생성 및 활성화
env_name="myenv"

# Conda 초기화 (필수!)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda 환경이 존재하지 않으면 생성
if ! conda info --envs | grep -q "$env_name"; then
    echo "[INFO] 가상환경 '$env_name' 생성 중..."
    conda create -y -n "$env_name" python=3.10
fi

# Conda 환경 활성화
conda activate "$env_name"


## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    filename="${file%.py}"  # e.g. 2243
    input_path="../input/${filename}_input"
    output_path="../output/${filename}_output"

    if [[ -f "$input_path" ]]; then
        echo "[INFO] $file 실행 중 (입력: $input_path)"
        python "$file" < "$input_path" > "$output_path"
        echo "[INFO] 결과 저장 완료: $output_path"
    else
        echo "[WARNING] 입력 파일 없음: $input_path → 스킵"
    fi

done

# mypy 테스트 실행 및 mypy_log.txt 저장
echo "[INFO] mypy 실행 중..."
mypy . --disable-error-code var-annotated --disable-error-code assignment > ../mypy_log.txt
echo "[INFO] mypy 결과가 mypy_log.txt에 저장됨"

# conda.yml 파일 생성
conda env export > ../conda.yml
echo "[INFO] conda.yml 저장 완료"

# 가상환경 비활성화
conda deactivate
echo "[INFO] 가상환경 비활성화 완료"
