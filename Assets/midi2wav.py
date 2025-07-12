#!/usr/bin/env python3
import subprocess, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import freeze_support   # ★ 추가

# ── 파라미터 설정 (필요 시 인자 파싱으로 교체) ──────────────────
HOME = Path(os.path.expanduser("~"))
K = 0 # 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f; these characters consist of MidiCaps's dir 

INPUT_DIR  = Path(fr"{HOME}/maap.github.io/Assets/midicaps_mid/lmd_full/{K}") 
OUTPUT_DIR = Path(fr"{HOME}/maap.github.io/Assets/lmd_full_audio") # After extract *.npy from *.mid in midicaps_mid/lmd_full/k
                                                                   #, All of *.wav should be deleted in OUTPUT_DIR
SOUNDFONT  = Path(fr"{HOME}/SoundFonts/FluidR3_GM.sf2") # Required
SAMPLE_SR  = 48000 # To change into better quallity of *.wav
MAX_WORKERS = min(os.cpu_count(), 8)
# ────────────────────────────────────────────────────────────────

def render_one(midi_file: Path) -> str:
    wav_path = OUTPUT_DIR / midi_file.with_suffix(".wav").name
    cmd = [
        "fluidsynth",
        "-ni",
        "-F", str(wav_path),
        "-r", str(SAMPLE_SR),
        SOUNDFONT,
        str(midi_file),
    ]
    subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    return wav_path.name

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    midi_files = sorted(INPUT_DIR.glob("*.mid")) + sorted(INPUT_DIR.glob("*.midi"))
    if not midi_files:
        sys.exit(f"[!] No MIDI files found in {INPUT_DIR}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex, \
         tqdm(total=len(midi_files), desc="Rendering", unit="file") as bar:
        futures = {ex.submit(render_one, m): m for m in midi_files}
        for fut in as_completed(futures):
            bar.set_postfix_str(fut.result())
            bar.update(1)

    print(f"\n✅  Done → WAVs saved to {OUTPUT_DIR}")

# ── Windows spawn 안전 가드 ──────────────────────────────────────
if __name__ == "__main__":
    freeze_support()      # PyInstaller 등으로 “실행 파일” 만들 때도 필수
    main()
