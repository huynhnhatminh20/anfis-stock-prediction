import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
from typing import Tuple

class StockDataset(Dataset):
    """Dataset tùy chỉnh cho dữ liệu chứng khoán."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class StockMLP(nn.Module):
    """Mạng Nơ-ron nhiều lớp (MLP) dự báo giá chứng khoán."""
    def __init__(self, input_size: int, hidden1: int = 64, hidden2: int = 32):
        super(StockMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_mlp_with_early_stopping(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 100, 
    lr: float = 0.001,
    patience: int = 10
) -> Tuple[nn.Module, list, list]:
    """Huấn luyện mô hình có tích hợp Early Stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_weights = None
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping kích hoạt tại epoch {epoch+1}!")
                break
                
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, train_losses, val_losses


