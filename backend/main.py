from huggingface_hub import HfApi

def get_under_10b_models(limit=10):
    api = HfApi()
    
    # Các từ khóa thường xuất hiện trong tên của các mô hình dưới 10B
    size_keywords = ["-1b", "-3b", "-7b", "-8b", "-9b", "1.5b", "3.8b"]
    
    print("Đang truy xuất API của Hugging Face...")
    
    # Lấy danh sách các mô hình text-generation, sắp xếp theo lượt tải nhiều nhất
    models = api.list_models(
        filter="text-generation",
        sort="downloads",
        limit=300  # Lấy dư ra để bù trừ khi lọc
    )
    
    under_10b_models = []
    
    for model in models:
        model_id_lower = model.id.lower()
        
        # Bỏ qua các bản quantize (GGUF, AWQ, GPTQ) để lấy model gốc (base/instruct)
        if any(q in model_id_lower for q in ["gguf", "awq", "gptq", "exl2"]):
            continue
            
        # Kiểm tra xem tên model có chứa từ khóa kích thước < 10B không
        if any(keyword in model_id_lower for keyword in size_keywords):
            under_10b_models.append({
                "Tên Model": model.id,
                "Lượt tải": model.downloads,
                "Likes": model.likes
            })
            
        # Dừng lại khi đã gom đủ số lượng yêu cầu
        if len(under_10b_models) >= limit:
            break
            
    return under_10b_models

if __name__ == "__main__":
    top_models = get_under_10b_models(limit=10)
    
    print(f"\n--- Top {len(top_models)} Mô hình dưới 10B phổ biến ---")
    print("-" * 50)
    for i, m in enumerate(top_models, 1):
        print(f"{i:2d}. {m['Tên Model']}")
        print(f"    Lượt tải: {m['Lượt tải']:,} | Likes: {m['Likes']:,}")
        print("-" * 50)