import argparse
import json
import os
import pandas as pd
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader

# Lấy đường dẫn gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.arima_model import build_and_train_arima, predict_arima, save_arima_model 
from src.evaluation.evaluator import Evaluator 
from src.models.mlp_model import train_mlp_with_early_stopping, StockMLP, StockDataset
from src.models.anfis_model import ANFIS
from src.models.anfis_train import AnfisTrainer
from src.config import AnfisConfig

def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm các mô hình dự báo.") 
    parser.add_argument('--model', type=str, required=True, choices=['arima', 'mlp', 'anfis', 'all']) 
    args = parser.parse_args()

    print(f"🚀 Bắt đầu thực nghiệm với lựa chọn: {args.model.upper()}")
    results = {}

    # --- BƯỚC 1: LẤY DỮ LIỆU CHUNG ---
    print("📂 Đang tải dữ liệu từ Data Pipeline...") 
    train_df = pd.read_csv('data/processed/VNM_train.csv')
    test_df = pd.read_csv('data/processed/VNM_test.csv')
    val_df = pd.read_csv('data/processed/VNM_val.csv')

    y_train = train_df['close']
    y_test = test_df['close']
    y_val = val_df['close']

    # Lọc bỏ cột chuỗi và cột mục tiêu
    X_train_df = train_df.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')
    X_test_df = test_df.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')
    X_val_df = val_df.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')

    # Chuyển sang Tensor
    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # --- BƯỚC 2: CHẠY CÁC KỊCH BẢN ---
    os.makedirs('models', exist_ok=True)

    # 1. ARIMA
    if args.model in ['arima', 'all']:
        print("\n--- 📈 Đang huấn luyện ARIMA ---") 
        model_arima = build_and_train_arima(y_train) 
        y_pred_arima, _ = predict_arima(model_arima, len(y_test)) 
        results['ARIMA'] = Evaluator.calculate_metrics(y_test.values, y_pred_arima) 
        save_arima_model(model_arima) 
        print(f"✅ Kết quả ARIMA: {results['ARIMA']}")

    # 2. MLP
    if args.model in ['mlp', 'all']:
        print("\n--- 🧠 Đang huấn luyện MLP ---")
        train_ds = StockDataset(X_train, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(StockDataset(X_val, y_val_tensor), batch_size=32, shuffle=False)

        mlp_model = StockMLP(input_size=X_train.shape[1])
        mlp_model, _, _ = train_mlp_with_early_stopping(mlp_model, train_loader, val_loader)

        mlp_model.eval()
        with torch.no_grad():
            y_pred_mlp = mlp_model(X_test).numpy()

        results['MLP'] = Evaluator.calculate_metrics(y_test.values, y_pred_mlp.flatten())
        torch.save(mlp_model.state_dict(), 'models/mlp_model.pth')
        print(f"✅ Kết quả MLP: {results['MLP']}")
    
    # 3. ANFIS
    if args.model in ['anfis', 'all']:
        print("\n--- 🔗 Đang huấn luyện ANFIS (Hybrid Learning) ---") 
        # Chọn Lọc: Giữ lại MA5 (4), RSI14 (6), MACD (7) để tránh nổ ma trận LSE
        selected_features = [4, 6, 7] 
        
        X_train_anfis = X_train[:, selected_features]
        X_test_anfis = X_test[:, selected_features]
        X_val_anfis = X_val[:, selected_features]

        # Ép Min-Max cục bộ chống Underflow cho Gaussian MF
        x_min = X_train_anfis.min(dim=0, keepdim=True)[0]
        x_max = X_train_anfis.max(dim=0, keepdim=True)[0]
        x_range = (x_max - x_min).clamp_min(1e-8)

        X_train_anfis = (X_train_anfis - x_min) / x_range
        X_test_anfis = (X_test_anfis - x_min) / x_range
        X_val_anfis = (X_val_anfis - x_min) / x_range

        config = AnfisConfig(
            epochs=50, 
            learning_rate=0.01,
            batch_size=64,
            use_hybrid_learning=True, 
            patience=10,
            random_seed=42,
            min_delta=1e-5,
            l2_weight_decay=1e-4
        )
        
        anfis_model = ANFIS(input_dim=len(selected_features), num_memberships=2)
        trainer = AnfisTrainer(config=config)
        
        artifacts = trainer.fit(
            model=anfis_model, train_features=X_train_anfis, train_targets=y_train_tensor,
            val_features=X_val_anfis, val_targets=y_val_tensor
        )
        
        y_pred_anfis = trainer.predict(artifacts.model, X_test_anfis).numpy()
        results['ANFIS'] = Evaluator.calculate_metrics(y_test.values, y_pred_anfis.flatten())
        trainer.save_model(artifacts.model, 'models/anfis_model.pth')
        print(f"✅ Kết quả ANFIS: {results['ANFIS']}")

    # --- BƯỚC 3: LƯU KẾT QUẢ ---
    os.makedirs('results', exist_ok=True) 
    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False) 
        
    print(f"\n🎉 Đã hoàn thành thực nghiệm toàn bộ hệ thống.")

if __name__ == "__main__":
    main()