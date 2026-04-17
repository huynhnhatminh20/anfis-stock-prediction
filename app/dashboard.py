import sys
import os
import torch
import joblib 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# Lấy đường dẫn gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.arima_model import predict_arima
from src.models.mlp_model import StockMLP
from src.models.anfis_model import ANFIS
from src.visualization.visualizer import Visualizer

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (CSS & THEME)
# ==========================================
st.set_page_config(page_title="ANFIS Stock Pro", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Helvetica Neue', Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stPlotlyChart { border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background: white; padding: 10px; }
    div.stButton > button:first-child { 
        background-color: #007bff; color: white; border-radius: 10px; 
        width: 100%; font-weight: bold; border: none; height: 3em;
    }
    div.stButton > button:first-child:hover { background-color: #0056b3; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HÀM LOAD TÀI NGUYÊN & DỮ LIỆU
# ==========================================
@st.cache_resource 
def load_models():
    models = {"arima": None, "mlp": None, "anfis": None}
    try:
        if os.path.exists("models/arima_model.pkl"):
            models["arima"] = joblib.load("models/arima_model.pkl")
        if os.path.exists("models/mlp_model.pth"):
            m = StockMLP(input_size=13) 
            m.load_state_dict(torch.load("models/mlp_model.pth", map_location='cpu', weights_only=True))
            m.eval(); models["mlp"] = m
        if os.path.exists("models/anfis_model.pth"):
            a = ANFIS(input_dim=3, num_memberships=2)
            a.load_state_dict(torch.load("models/anfis_model.pth", map_location='cpu', weights_only=True))
            a.eval(); models["anfis"] = a
        return models
    except Exception as e:
        st.error(f"Lỗi load models: {e}"); return models

def load_stock_data(ticker):
    # Đường dẫn tới file thô (Raw) để lấy giá tiền thật
    # Bạn hãy kiểm tra lại đường dẫn này có đúng là 'data/raw/' không nhé
    raw_path = f"data/raw/VNM_2020-01-01_2026-03-30_1d.csv" 
    test_path = f"data/processed/{ticker}_test.csv"
    
    if not os.path.exists(test_path):
        st.error(f"Không tìm thấy file test tại {test_path}")
        return None
        
    df_test = pd.read_csv(test_path)
    
    # 1. TÌM GIÁ TRỊ THỰC TỪ FILE RAW
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        # Lấy Min/Max từ file thô (ví dụ: 73.04 và 74.43)
        min_p = df_raw['close'].min()
        max_p = df_raw['close'].max()
        is_scaled = False # File raw chắc chắn là tiền thật
    else:
        # Nếu không có file raw, ta dùng tạm con số mặc định để demo
        st.warning(f"Không tìm thấy file raw tại {raw_path}, đang dùng giá giả lập.")
        min_p, max_p = 60.0, 90.0
        is_scaled = True

    # 2. XỬ LÝ NGÀY THÁNG
    date_col = 'date' if 'date' in df_test.columns else df_test.columns[0]
    dates = pd.to_datetime(df_test[date_col], errors='coerce').ffill().bfill().tolist()
    
    return {
        "df_test": df_test,
        "min": min_p,
        "max": max_p,
        "p_col": 'close',
        "dates": dates
    }

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
def main():
    st.title("📈 ANFIS Stock Forecasting Dashboard")
    st.write("**Nhóm:** Hải, Minh, Nhân, Minh | **Dự báo giá trị thực (VNĐ)**")
    
    # Sidebar
    st.sidebar.header("⚙️ Cấu hình")
    ticker = st.sidebar.selectbox("Mã Cổ phiếu", ["VNM", "ACB", "VIC", "HPG", "FPT"])
    model_choice = st.sidebar.radio("Mô hình", ["ANFIS (Lai ghép)", "ARIMA (Thống kê)", "MLP (Deep Learning)"])
    forecast_days = st.sidebar.slider("Số ngày dự báo", 1, 10, 3)
    show_arch = st.sidebar.checkbox("Hiển thị kiến trúc ANFIS")
    run_btn = st.sidebar.button("🚀 THỰC THI DỰ BÁO")

    if show_arch:
        with st.expander("Sơ đồ kiến trúc ANFIS 5 lớp", expanded=True):
            st.pyplot(Visualizer.plot_anfis_architecture())

    if run_btn:
        data = load_stock_data(ticker)
        if data is None:
            st.error(f"Thiếu dữ liệu train/test cho {ticker} trong thư mục data/processed/"); return

        with st.spinner('Đang tính toán dự báo...'):
            df_test = data["df_test"]
            dates = data["dates"]
            prices_scaled = df_test[data["p_col"]].tolist()
            
            models = load_models()
            preds_scaled = [] 

            # --- DỰ BÁO ---
            if model_choice == "ARIMA (Thống kê)":
                if models["arima"]:
                    p, _ = predict_arima(models["arima"], forecast_days)
                    preds_scaled = p.tolist()
            
            elif model_choice == "MLP (Deep Learning)":
                if models["mlp"]:
                    X = df_test.select_dtypes(include=[np.number]).drop(columns=['close','Close'], errors='ignore').tail(1)
                    with torch.no_grad():
                        out = models["mlp"](torch.tensor(X.values, dtype=torch.float32)).item()
                        preds_scaled = [out * (1 + np.random.normal(0, 0.002)) for _ in range(forecast_days)]

            elif model_choice == "ANFIS (Lai ghép)":
                if models["anfis"]:
                    cols = ['ma5', 'rsi14', 'macd']
                    # Scale input theo Min-Max của tập Train
                    df_train = pd.read_csv(f"data/processed/{ticker}_train.csv")
                    x_min, x_max = df_train[cols].min().values, df_train[cols].max().values
                    x_range = np.maximum(x_max - x_min, 1e-8)
                    X_input = (df_test[cols].tail(1).values - x_min) / x_range
                    with torch.no_grad():
                        out = models["anfis"](torch.tensor(X_input, dtype=torch.float32)).item()
                        preds_scaled = [out * (1 + np.random.normal(0, 0.001)) for _ in range(forecast_days)]

            # --- ĐẢO CHUẨN HÓA SANG TIỀN THẬT ---
            # Công thức: Thực = Scaled * (Max - Min) + Min
            diff = data["max"] - data["min"]
            final_preds = [p * diff + data["min"] for p in preds_scaled]
            actual_real = [p * diff + data["min"] for p in prices_scaled]

            # --- VẼ BIỂU ĐỒ ---
            fig = go.Figure()
            # Thực tế
            fig.add_trace(go.Scatter(x=dates[-30:], y=actual_real[-30:], name='Thực tế', line=dict(color='#2E86C1', width=3)))
            
            # Dự báo
            f_x = [dates[-1]] + [dates[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            f_y = [actual_real[-1]] + final_preds
            fig.add_trace(go.Scatter(x=f_x, y=f_y, name='Dự báo', line=dict(color='#E74C3C', width=3, dash='dash')))
            
            # Nón Monte Carlo
            for _ in range(15):
                path = [actual_real[-1]]
                for j in range(forecast_days):
                    noise = np.random.normal(0, 0.015 * np.sqrt(j + 1))
                    path.append(f_y[j+1] * (1 + noise))
                fig.add_trace(go.Scatter(x=f_x, y=path, mode='lines', line=dict(color='gray', width=1), opacity=0.1, showlegend=False))

            fig.update_layout(template="plotly_white", title=f"Giá {ticker} (VNĐ)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # Bảng số liệu
            st.table(pd.DataFrame({
                "Ngày": [(dates[-1] + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, forecast_days+1)], 
                "Giá dự báo (VNĐ)": [f"{p:,.2f}" for p in final_preds]
            }))

if __name__ == "__main__":
    main()