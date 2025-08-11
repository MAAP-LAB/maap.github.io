"""
Google Colab용 BLAP LoRA 훈련 스크립트
Colab 노트북에서 직접 실행 가능
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """필요한 패키지들 설치"""
    print("📦 Installing required packages...")
    
    packages = [
        "peft",
        "transformers==4.30.0",
        "torch",
        "matplotlib",
        "numpy",
        "tqdm"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed!")

def run_lora_training():
    """LoRA 훈련 실행"""
    print("🚀 Starting BLAP LoRA Fine-tuning in Colab")
    print("=" * 50)
    
    # Colab 환경에서의 기본 경로 설정
    base_dir = "/content"  # Colab의 기본 작업 디렉토리
    
    # 파일 경로들 (Colab에서 업로드한 위치에 맞게 수정 필요)
    config = {
        "blap_checkpoint": f"{base_dir}/blap/checkpoint/checkpoint.ckpt",
        "clamp3_weights": f"{base_dir}/clamp3/code/weights.pth", 
        "config_path": f"{base_dir}/blap/checkpoint/config.json",
        "train_json": f"{base_dir}/Adapters/FinetuneMusicQA_npy.json",
        "val_json": f"{base_dir}/Adapters/EvalMusicQA_npy.json",
        "save_dir": f"{base_dir}/lora_checkpoints",
        
        # LoRA 하이퍼파라미터
        "lora_r": 16,
        "lora_alpha": 32, 
        "lora_dropout": 0.1,
        
        # 훈련 하이퍼파라미터
        "batch_size": 4,  # Colab GPU 메모리에 맞게 작게 설정
        "epochs": 12,
        "lr": 2e-4,
        "accumulation_steps": 8,  # 작은 배치 사이즈 보상
    }
    
    # 명령어 구성
    cmd = [
        sys.executable, f"{base_dir}/Adapters/blap_lora_trainer.py",
        "--blap_checkpoint", config["blap_checkpoint"],
        "--clamp3_weights", config["clamp3_weights"],
        "--config_path", config["config_path"],
        "--train_json", config["train_json"],
        "--val_json", config["val_json"],
        "--batch_size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--accumulation_steps", str(config["accumulation_steps"]),
        "--lora_r", str(config["lora_r"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--lora_dropout", str(config["lora_dropout"]),
        "--save_dir", config["save_dir"]
    ]
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 훈련 실행
    try:
        subprocess.run(cmd, check=True)
        print("🎉 LoRA Fine-tuning completed successfully!")
        print(f"Check results in: {config['save_dir']}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        print("Please check the file paths and GPU availability.")

if __name__ == "__main__":
    # 패키지 설치
    install_requirements()
    
    # LoRA 훈련 실행
    run_lora_training()