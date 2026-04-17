import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, Union



class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # Đảm bảo đầu vào là mảng 1 chiều
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 1. Tính MAE và RMSE bình thường
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 2. Xử lý triệt để MAPE: Bỏ qua các điểm y_true = 0 (do Min-Max sinh ra)
        # Chỉ tính MAPE trên những ngày có y_true > 0.001
        mask = y_true > 0.001
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0  # Hoặc np.nan tùy bạn
            
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    @staticmethod
    def plot_predictions(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], title: str = "Actual vs Predicted"):
   
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Predicted', linestyle='--')
        ax.set_title(title)
        ax.legend()
        plt.close(fig) # Đóng fig lại để tránh rò rỉ bộ nhớ khi chạy nhiều lần
        return fig

    @staticmethod
    def residual_plot(residuals: Union[np.ndarray, list]):
        """Vẽ phần dư và đồ thị tự tương quan ACF."""
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(residuals)
        ax[0].set_title("Residuals")
        plot_acf(residuals, ax=ax[1])
        plt.close(fig)
        return fig
