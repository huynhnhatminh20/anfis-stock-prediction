import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, Union

class Evaluator:
    @staticmethod
    def calculate_metrics(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> Dict[str, float]:
        """Tính toán các chỉ số đánh giá mô hình[cite: 11, 79]."""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }

    @staticmethod
    def plot_predictions(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], title: str = "Actual vs Predicted"):
        """Vẽ biểu đồ dự báo và trả về đối tượng Figure[cite: 79, 91]."""
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