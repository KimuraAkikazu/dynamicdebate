from pathlib import Path
from huggingface_hub import hf_hub_download

# 修正：ダウンロードするモデルを変更
REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"  # 必要に応じてファイル名を修正

# REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
# FILENAME = "llama-2-7b-chat.Q4_K_M.gguf"


BASE_DIR = Path(__file__).parent.resolve()
MODEL_SAVE_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_SAVE_DIR / FILENAME

def get_model_path():
    """
    モデルをダウンロードし、ローカルパスを返す。
    """
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_SAVE_DIR,
        local_dir_use_symlinks=False,
        force_filename=FILENAME,
        resume_download=False
    )
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    print(f"モデル保存先: {MODEL_PATH}")
    return str(MODEL_PATH)  # 文字列に変換して返す
