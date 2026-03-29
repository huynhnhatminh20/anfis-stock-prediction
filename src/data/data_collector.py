from vnstock import *
import pandas as pd
import os
def load_stock_data(symbol="VNM", start="2023-01-01", end="2024-12-31"):
    stock = Vnstock().stock(symbol=symbol, source="VCI")
    df = stock.quote.history(start=start, end=end, interval="1D")
    return df
def save_data(df, symbol):
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{symbol}.csv"
    df.to_csv(path, index=False)
    print(f"Đã lưu dữ liệu vào {path}")
if __name__ == "__main__":
    df = load_stock_data()
    print(df.head())
    save_data(df, "VNM")