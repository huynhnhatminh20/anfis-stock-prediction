import optuna
import torch
import torch.nn as nn
from src.models.mlp_model import StockMLP

def objective_mlp(trial, train_loader, val_loader, input_size):
    hidden1 = trial.suggest_int("hidden1", 32, 256, step=32)
    hidden2 = trial.suggest_int("hidden2", 16, 128, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    model = StockMLP(input_size=input_size, hidden1=hidden1, hidden2=hidden2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train nhanh 30 epochs để đánh giá tiềm năng của bộ tham số
    model.train()
    for _ in range(30):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_loss += criterion(model(X_val), y_val).item()
            
    return (val_loss / len(val_loader)) ** 0.5 # Trả về RMSE

def tune_mlp(train_loader, val_loader, input_size, n_trials=20):
    print(" Bắt đầu dò tìm siêu tham số với Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_mlp(trial, train_loader, val_loader, input_size), n_trials=n_trials)
    print(f"Tham số tốt nhất: {study.best_params}")
    return study.best_params




# ==========================================
# KHU VỰC DÀNH CHO THÀNH VIÊN A (TV1) GHÉP CODE
# ==========================================
def objective_anfis(trial, train_data, val_data):
    """Hàm mục tiêu tối ưu cho ANFIS (TV1 sẽ code chi tiết)."""
    # num_mfs = trial.suggest_int("num_mfs", 2, 5)
    # mf_type = trial.suggest_categorical("mf_type", ["gaussmf", "gbellmf"])
    # lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    # ... logic train và trả về RMSE ...
    pass

def tune_anfis(train_data, val_data, n_trials=20):
    """Chạy Optuna study cho ANFIS (TV1 sẽ code)."""
    pass