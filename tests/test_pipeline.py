"""End-to-End Integration Tests for the entire ML Pipeline."""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import các modules của dự án
from src.data.data_pipeline import StockDataPipeline
from src.models.mlp_model import StockMLP, StockDataset, train_mlp_with_early_stopping
from src.evaluation.evaluator import Evaluator
from src.config import AnfisConfig
from src.models.anfis_model import ANFIS
from src.models.anfis_train import AnfisTrainer

def _make_sample_df(n: int = 150) -> pd.DataFrame:
    """Tạo dữ liệu chứng khoán giả lập để chạy test nhanh."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.linspace(100, 150, n)
    noise = 6 * np.sin(np.linspace(0, 20, n))
    close = base + noise

    df = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, n),
        }
    )
    return df

def test_end_to_end_mlp_pipeline(tmp_path: Path) -> None:
    """Test luồng chạy từ Data thô -> Xử lý -> Train MLP -> Tính Metrics."""
    # 1. Chuẩn bị dữ liệu thô
    df_raw = _make_sample_df(150)
    
    # 2. Chạy Data Pipeline (Dùng tmp_path của pytest để không ghi rác ra thư mục thật)
    pipeline = StockDataPipeline(processed_dir=tmp_path)
    data_out = pipeline.fit_transform_from_df(df_raw, symbol="TEST", save_processed=False)
    
    # Tách X, y
    # Tách X, y
    X_train_df = data_out.train.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')
    X_val_df = data_out.val.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')
    X_test_df = data_out.test.select_dtypes(include=[np.number]).drop(columns=['close', 'Close'], errors='ignore')

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train = torch.tensor(data_out.train['close'].values, dtype=torch.float32).view(-1, 1)
        
    X_val = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val = torch.tensor(data_out.val['close'].values, dtype=torch.float32).view(-1, 1)
        
    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test_actual = data_out.test['close'].values

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=16)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=16)   
        
    # 3. Train MLP (Chạy siêu nhanh với 2 epochs để test code, không test độ chính xác)
    model = StockMLP(input_size=X_train.shape[1], hidden1=16, hidden2=8)

    # Lưu ý: có thể cần mock hoặc set epochs nhỏ trong hàm train_mlp_with_early_stopping
    model, t_loss, v_loss = train_mlp_with_early_stopping(model, train_loader, val_loader)
    
    # 4. Dự báo và đảo chuẩn hóa
    model.eval()
    with torch.no_grad():
        y_pred_scaled = pd.Series(model(X_test).numpy().flatten())
    y_test_actual = data_out.test['close']
    y_pred_real = pipeline.inverse_transform_close(y_pred_scaled)
    y_test_real = pipeline.inverse_transform_close(y_test_actual)
    
    # 5. Đánh giá (Evaluator)
    metrics = Evaluator.calculate_metrics(y_test_real, y_pred_real)
    
    # Assertions: Đảm bảo toàn bộ pipeline sinh ra kết quả hợp lệ
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] >= 0
    assert metrics["MAPE"] >= 0
    assert len(y_pred_real) == len(y_test_real)

def test_evaluator_handles_zero_values() -> None:
    """Test để đảm bảo Evaluator không bị lỗi chia cho 0 khi tính MAPE."""
    y_true = np.array([0.0, 100.0, 0.0, 50.0])
    y_pred = np.array([10.0, 90.0, 5.0, 55.0])
    
    metrics = Evaluator.calculate_metrics(y_true, y_pred)
    
    # Hàm tính MAPE phải bỏ qua các giá trị 0.0 của y_true
    # Các cặp hợp lệ: (100, 90) -> error 10%, (50, 55) -> error 10%. Mean = 10%
    assert np.isclose(metrics["MAPE"], 10.0)
    assert metrics["MAE"] > 0