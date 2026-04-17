import optuna
import torch
import torch.nn as nn
from src.models.mlp_model import StockMLP
from src.models.anfis_model import ANFIS

try:
    from src.models.anfis_train import AnfisTrainer
    from src.config import AnfisConfig
except ImportError:
    pass # Bỏ qua lỗi import nếu đang test rời


# ==========================================
# KHU VỰC DÀNH CHO MLP
# ==========================================
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
    print(" Bắt đầu dò tìm siêu tham số MLP với Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_mlp(trial, train_loader, val_loader, input_size), n_trials=n_trials)
    print(f"Tham số tốt nhất cho MLP: {study.best_params}")
    return study.best_params


# ==========================================
# KHU VỰC DÀNH ANFIS
# ==========================================
def objective_anfis(trial, train_features, train_targets, val_features, val_targets):
    """Hàm mục tiêu tối ưu cho ANFIS."""
    input_dim = train_features.shape[1]
    
    # 1. Suggest tham số
    # CẢNH BÁO: Giới hạn num_memberships ở mức 2-4. 
    # ANFIS bị "lời nguyền số chiều" (Curse of Dimensionality). Nếu để >= 5 với 13 features, RAM sẽ nổ tung!
    num_memberships = trial.suggest_int("num_memberships", 2, 3)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # 2. Khởi tạo cấu hình (Train nhanh 15 epochs để đánh giá tiềm năng)
    config = AnfisConfig(
        epochs=15, 
        learning_rate=lr,
        batch_size=batch_size,
        use_hybrid_learning=True, # Luôn bật Hybrid Learning để hội tụ nhanh
        patience=5,
        random_seed=42
    )
    
    # 3. Khởi tạo mô hình và Trainer
    model = ANFIS(input_dim=input_dim, num_memberships=num_memberships)
    trainer = AnfisTrainer(config=config)
    
    # 4. Huấn luyện (Fit)
    artifacts = trainer.fit(
        model=model,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets
    )
    
    # 5. Lấy best validation loss (là MSE) và trả về RMSE
    best_val_mse = artifacts.history.best_val_loss
    return best_val_mse ** 0.5

def tune_anfis(train_features, train_targets, val_features, val_targets, n_trials=15):
    """Chạy Optuna study cho ANFIS."""
    print("Bắt đầu dò tìm siêu tham số ANFIS với Optuna...")
    study = optuna.create_study(direction="minimize")
    
    # Dùng lambda để truyền thêm tensor dữ liệu vào hàm objective
    study.optimize(
        lambda trial: objective_anfis(trial, train_features, train_targets, val_features, val_targets), 
        n_trials=n_trials
    )
    
    print(f"Tham số tốt nhất cho ANFIS: {study.best_params}")
    return study.best_params