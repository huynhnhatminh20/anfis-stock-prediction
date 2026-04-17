import pmdarima as pm
import torch
import numpy as np
import pandas as pd
import joblib
import os
from typing import Union, Tuple

def build_and_train_arima(train_data: Union[pd.Series, np.ndarray]) -> pm.arima.ARIMA:
    """
    Tự động dò tìm tham số (p, d, q) bằng AIC và huấn luyện mô hình ARIMA[cite: 69, 76].
    """
    print("⏳ Đang tự động dò tìm tham số ARIMA bằng auto_arima...")
    model = pm.auto_arima(
        train_data,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        m=1,              
        seasonal=False,   
        d=None,           
        trace=True,       
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True,
        information_criterion='aic'
    )
    print(f"✅ Mô hình ARIMA tốt nhất: {model.summary()}")
    return model

def predict_arima(model: pm.arima.ARIMA, n_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dự báo n bước thời gian tiếp theo và trả về khoảng tin cậy
    """
    predictions, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    return np.array(predictions), np.array(conf_int)

def save_arima_model(model: pm.arima.ARIMA, folder: str = "models", filename: str = "arima_model.pkl"):
    """
    Lưu mô hình ARIMA vào file để Dashboard có thể sử dụng.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    joblib.dump(model, path)
   
    print(f"💾 Đã lưu mô hình tại: {path}")

def load_arima_model(folder: str = "models", filename: str = "arima_model.pkl") -> pm.arima.ARIMA:
    """
    Tải mô hình ARIMA từ file.
    """
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"❌ Không tìm thấy file tại {path}")
        return None