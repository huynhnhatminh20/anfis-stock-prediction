import pandas as pd
import os
import json

class ReportGenerator:
    """Module tự động tạo báo cáo kết quả thực nghiệm."""

    @staticmethod
    def generate_from_json(json_path: str = "results/metrics.json", output_dir: str = "results"):
        """Đọc file metrics.json từ TV3 và tự động xuất báo cáo."""
        if not os.path.exists(json_path):
            print(f"⚠️ Không tìm thấy file {json_path}. Vui lòng chạy experiment_runner.py trước!")
            return
            
        with open(json_path, 'r', encoding='utf-8') as f:
            metrics_dict = json.load(f)
            
        ReportGenerator.export_results(metrics_dict, output_dir)

    @staticmethod
    def export_results(metrics_dict: dict, output_dir: str = "results"):
        """Xuất DataFrame kết quả ra Excel và file .tex (LaTeX)."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Chuyển đổi từ Dictionary sang DataFrame
        df = pd.DataFrame.from_dict(metrics_dict, orient='index').reset_index()
        df.rename(columns={'index': 'Mô hình'}, inplace=True)
        
        # 1. Xuất Excel
        excel_path = os.path.join(output_dir, "model_comparison.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"✅ Đã xuất báo cáo Excel tại: {excel_path}")
        
        # 2. Xuất LaTeX cho quyển Luận văn (Cập nhật chuẩn Pandas 2.0+)
        latex_path = os.path.join(output_dir, "model_comparison.tex")
        
        # Dùng Styler thay vì df.to_latex() cũ để tránh Warning
        styler = df.style.format(precision=4)
        latex_str = styler.to_latex(
            caption="So sánh hiệu suất dự báo giữa ANFIS, ARIMA và MLP", 
            label="tab:model_comparison",
            hrules=True, # Tự động thêm đường kẻ ngang chuẩn format bảng khoa học (toprule, midrule, bottomrule)
            position="h!" 
        )
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        print(f"✅ Đã tạo mã LaTeX tại: {latex_path}")


if __name__ == "__main__":
    ReportGenerator.generate_from_json()
    ReportGenerator.export_results({
        {
    "ARIMA": {
        "MAE": 0.11266918519738145,
        "RMSE": 0.14580621436362992,
        "MAPE": 41.44819356551759
    },
    "MLP": {
        "MAE": 0.12504102750373555,
        "RMSE": 0.1769263594598919,
        "MAPE": 48.23142761291415
    },
    "ANFIS": {
        "MAE": 0.04738559692354933,
        "RMSE": 0.06523221125092087,
        "MAPE": 22.677317666655625
    }
        }
    })

    