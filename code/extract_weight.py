#!/usr/bin/env python
# save_as_safetensors.py
import torch, safetensors.torch as st, pathlib, os

# 변환할 체크포인트 목록
src_files = [
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_32_p_length_512_45.pth",
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_32_p_length_512_65_10.pth",
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_32_p_length_512_65_35.pth",
]

out_dir = pathlib.Path("weight_only")
out_dir.mkdir(exist_ok=True)

for src in src_files:
    src_path = pathlib.Path(src)
    print(f"→ 로드 중: {src_path}")
    ckpt = torch.load(src_path, map_location="cpu")        # weights_only=False (기본)

    # 1) optimizer/scheduler 제거 → 순수 파라미터만 추출
    if "state_dict" in ckpt:               # lightning 스타일
        state = ckpt["state_dict"]
    elif "model" in ckpt:                  # accelerate·deepspeed 스타일
        state = ckpt["model"]
    else:                                  # 순수 dict 타입
        state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    # 2) safetensors 저장
    dst = out_dir / (src_path.stem + ".safetensors")
    st.save_file(state, dst)
    print(f"   ↳ saved {dst}  ({os.path.getsize(dst)/1e9:.2f} GB)")
