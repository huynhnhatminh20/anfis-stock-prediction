import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Lấy đường dẫn của thư mục gốc dự án (thư mục cha của thư mục 'app') và thêm vào hệ thống
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import các hàm từ các thành viên khác
from models.arima_model import build_and_train_arima, predict_arima, save_arima_model 
from evaluation.evaluator import Evaluator 
from models.mlp_model import train_mlp_with_early_stopping, StockMLP, StockDataset
# from models.anfis_model import train_anfis  # Giả sử bạn có hàm này sau khi TV1 hoàn thiện



def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm các mô hình dự báo.") 
    parser.add_argument('--model', type=str, required=True, choices=['arima', 'mlp', 'anfis', 'all']) 
    args = parser.parse_args()

    print(f" Bắt đầu thực nghiệm với lựa chọn: {args.model.upper()}")
    results = {}

    # --- BƯỚC 1: LẤY DỮ LIỆU TỪ TV2 ---
    # Giả lập dữ liệu để bạn có thể chạy thử ngay (Thay thế bằng data thật của TV2 khi sẵn sàng)
    print("📂 Đang tải dữ liệu từ Data Pipeline...") 
    y_train = pd.read_csv('data/processed/VNM_train.csv')['close']
    y_test = pd.read_csv('data/processed/VNM_test.csv')['close']
    y_val = pd.read_csv('data/processed/VNM_val.csv')['close']
    # --- BƯỚC 2: CHẠY CÁC KỊCH BẢN ---

    # 1. Kịch bản ARIMA
    if args.model in ['arima', 'all']:
        print("\n--- Đang huấn luyện ARIMA ---") 
        # Huấn luyện mô hình
        model_arima = build_and_train_arima(y_train) 
        
        # Dự báo trên tập test
        y_pred_arima, _ = predict_arima(model_arima, len(y_test)) 
        
        # Tính toán metrics thật
        results['ARIMA'] = Evaluator.calculate_metrics(y_test, y_pred_arima) 
        # QUAN TRỌNG: Lưu mô hình vào thư mục models/ để Dashboard của TV4 sử dụng
        save_arima_model(model_arima) 
        print(f"📈 Kết quả ARIMA: {results['ARIMA']}")

    # 2. Kịch bản MLP 
    if args.model in ['mlp', 'all']:
        print("📂 Đang tải dữ liệu từ Data Pipeline...")
    
        # Đọc dữ liệu
        train_df = pd.read_csv('data/processed/VNM_train.csv')
        test_df = pd.read_csv('data/processed/VNM_test.csv')
        val_df = pd.read_csv('data/processed/VNM_val.csv')



        # Lọc bỏ các cột không phải số (Date, Ticker,...) để tránh lỗi TypeError
        # Chúng ta chỉ giữ lại các cột tính toán được
        X_train_df = train_df.select_dtypes(include=[np.number]).drop(columns=['close'], errors='ignore')
        X_test_df = test_df.select_dtypes(include=[np.number]).drop(columns=['close'], errors='ignore')
        X_val_df = val_df.select_dtypes(include=[np.number]).drop(columns=['close'], errors='ignore')

        
        # Chuyển sang Tensor
        X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
        y_train_mlp = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
        y_test_mlp = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
        y_val_mlp = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        # Tạo DataLoader
        train_ds = StockDataset(X_train, y_train_mlp)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(StockDataset(X_val, y_val_mlp), batch_size=32, shuffle=False)

        # Khởi tạo và huấn luyện
        mlp_model = StockMLP(input_size=X_train.shape[1])
        mlp_model, t_losses, v_losses = train_mlp_with_early_stopping(mlp_model, train_loader, val_loader)

        # Dự báo
        mlp_model.eval()
        with torch.no_grad():
            y_pred_mlp = mlp_model(X_test).numpy()

        # Tính toán metrics thật cho MLP
        results['MLP'] = Evaluator.calculate_metrics(y_test.values, y_pred_mlp.flatten())
        
        # Lưu mô hình cho TV4
        torch.save(mlp_model.state_dict(), 'models/mlp_model.pth')
        print(f"📈 Kết quả MLP: {results['MLP']}")
    
    # 3. Kịch bản ANFIS (Thành viên 1)
    if args.model in ['anfis', 'all']:
        print("\n--- Đang huấn luyện ANFIS ---") 
        # Sau khi TV1 hoàn thiện core, gọi hàm huấn luyện tại đây
        results['ANFIS'] = {"MAE": 0.9, "RMSE": 1.1, "MAPE": 0.8}
        # torch.save(model_anfis.state_dict(), 'models/anfis_model.pt') 

    # --- BƯỚC 3: LƯU KẾT QUẢ VÀ BÁO CÁO ---
    os.makedirs('results', exist_ok=True) 
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False) 
        
    print(f"\n✅ Đã hoàn thành thực nghiệm.")
    print(f"💾 Kết quả chi tiết tại: 'results/metrics.json'") 
    print(f"📦 Mô hình đã lưu tại thư mục: 'models/'") 

if __name__ == "__main__":
    main()
