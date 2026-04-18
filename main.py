import os
import subprocess
import shutil
import re
import io
import sys
import time
import threading
import gradio as gr
from huggingface_hub import HfApi, snapshot_download
from dotenv import load_dotenv

# --- Khởi tạo và Load Token ---
# Đọc các biến môi trường từ file .env
load_dotenv()

# Lấy token từ .env (hỗ trợ cả chữ hoa và thường theo thói quen của bạn)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_token")

def get_gpu_vram_gb():
    """Đọc VRAM từ GPU NVIDIA."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return round(int(output.strip().split('\n')[0]) / 1024, 1)
    except:
        return 0

def get_available_categories(vram_gb):
    """Chỉ trả về các phân khúc mà máy có thể gánh được (ước tính cho bản 4-bit)."""
    all_cats = [
        "1. Dưới 3B (Mọi cấu hình)",
        "2. Từ 3B - 8B (Yêu cầu ~6GB+ VRAM)",
        "3. Từ 8B - 14B (Yêu cầu ~12GB+ VRAM)",
        "4. Trên 14B (Yêu cầu ~24GB+ VRAM)"
    ]
    
    if vram_gb < 6:
        return all_cats[:1]
    elif vram_gb < 12:
        return all_cats[:2]
    elif vram_gb < 20:
        return all_cats[:3]
    else:
        return all_cats

def get_models_by_size(size_category):
    """Lấy list model từ HF dựa trên size đã chọn, có dùng Token để fetch nhanh hơn."""
    if not size_category: return gr.update(choices=[])
    
    patterns = {
        "1. Dưới 3B": r'\b(0\.\db|1(\.\d)?b|2(\.\d)?b)\b',
        "2. Từ 3B - 8B": r'\b(3(\.\d)?b|4b|6b|7b|8b)\b',
        "3. Từ 8B - 14B": r'\b(9b|10b|11b|12b|13b|14b)\b',
        "4. Trên 14B": r'\b(32b|70b|72b|110b)\b'
    }
    
    current_pattern = next((p for k, p in patterns.items() if k in size_category), patterns["1. Dưới 3B"])
    
    # Truyền token vào API
    api = HfApi(token=HF_TOKEN)
    try:
        models = api.list_models(filter="text-generation", sort="downloads", limit=200)
        result = [m.id for m in models if re.search(current_pattern, m.id.lower()) and 
                  not any(q in m.id.lower() for q in ["gguf", "awq", "gptq"])]
        return gr.update(choices=result[:15], value=result[0] if result else None)
    except Exception as e:
        return gr.update(choices=[f"Lỗi kết nối HF: {str(e)}"])

def download_model_to_local(model_id):
    """Tải model và stream trực tiếp log Terminal lên Web UI"""
    if not model_id or "Lỗi" in model_id:
        yield "❌ Vui lòng chọn một model hợp lệ trước khi tải."
        return
    
    target_dir = os.path.join(os.getcwd(), "Models", model_id.replace("/", "--"))
    
    # 1. Tạo một bộ đệm để hứng log terminal
    log_buffer = io.StringIO()
    original_stderr = sys.stderr
    sys.stderr = log_buffer  # Bắt cóc luồng in lỗi (tqdm mặc định in ra stderr)
    
    # Biến để bắt lỗi từ bên trong thread
    download_error = []

    # 2. Định nghĩa tác vụ tải chạy ngầm
    def download_task():
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                token=HF_TOKEN
            )
        except Exception as e:
            download_error.append(str(e))
            print(f"\n❌ LỖI: {str(e)}", file=sys.stderr)

    # 3. Kích hoạt Thread để quá trình tải không làm đơ giao diện web
    thread = threading.Thread(target=download_task)
    thread.start()
    
    token_status = "🔒 (Dùng Token)" if HF_TOKEN else "🔓 (Ẩn danh)"
    
    # 4. Vòng lặp liên tục đọc log và gửi lên UI
    try:
        while thread.is_alive():
            raw_log = log_buffer.getvalue()
            if raw_log:
                # Xử lý ký tự \r của tqdm để bóc tách từng dòng riêng biệt
                lines = [line.strip() for line in raw_log.replace('\n', '\r').split('\r') if line.strip()]
                
                # Lấy 15 dòng log mới nhất để UI hiển thị mượt mà
                display_log = "\n".join(lines[-15:])
                yield f"⏳ Đang tải: {model_id} {token_status}\n\n[Terminal Log]:\n{display_log}"
            
            time.sleep(0.3) # Cập nhật UI mỗi 0.3s
            
        # Kiểm tra xem có lỗi xảy ra trong thread không
        thread.join()
        if download_error:
            yield f"❌ Lỗi khi tải:\n{download_error[0]}"
        else:
            yield f"✅ HOÀN TẤT! Model đã sẵn sàng tại:\n{target_dir}"
            
    finally:
        # 5. Luôn đảm bảo trả lại luồng stderr về mặc định cho hệ thống sau khi xong
        sys.stderr = original_stderr

# --- Khởi tạo UI ---
vram = get_gpu_vram_gb()
allowed_cats = get_available_categories(vram)

with gr.Blocks(title="LLM Manager", theme=gr.themes.Default()) as demo:
    gr.Markdown(f"# 🤖 LLM Download Manager\n**VRAM phát hiện:** {vram} GB")
    
    if not HF_TOKEN:
        gr.Warning("Chưa tìm thấy HF_TOKEN trong file .env. Tốc độ tải có thể bị giới hạn và không tải được Llama/Gemma.")
    
    with gr.Row():
        size_drop = gr.Dropdown(choices=allowed_cats, label="1. Chọn kích thước (Đã lọc theo GPU)", value=allowed_cats[-1])
        model_drop = gr.Dropdown(choices=[], label="2. Chọn Model cụ thể")
    
    download_btn = gr.Button("🚀 Tải Model về máy", variant="primary")
    
    # Textbox lớn hơn để chứa log
    status_output = gr.Textbox(
        label="💻 Terminal Output (Trực tiếp)", 
        interactive=False, 
        lines=12, 
        max_lines=15
    )

    size_drop.change(get_models_by_size, inputs=size_drop, outputs=model_drop)
    download_btn.click(download_model_to_local, inputs=model_drop, outputs=status_output)
    demo.load(get_models_by_size, inputs=size_drop, outputs=model_drop)

if __name__ == "__main__":
    os.makedirs("Models", exist_ok=True)
    demo.launch()