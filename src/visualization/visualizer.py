import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Visualizer:
    """Class chứa các hàm trực quan hoá cho đồ án."""

    @staticmethod
    def plot_learning_curve(train_losses: list, val_losses: list, save_path: str = None):
        """Vẽ đường cong học tập (Learning Curve) và trả về đối tượng Figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss (Lỗi huấn luyện)', color='#1f77b4', linewidth=2)
        ax.plot(val_losses, label='Validation Loss (Lỗi xác thực)', color='#ff7f0e', linewidth=2)
        
        ax.set_title('Đường cong học tập theo Epoch', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Loss (MSE / RMSE)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close(fig) # Đóng fig lại để tránh rò rỉ bộ nhớ khi chạy trên Streamlit
        return fig

    @staticmethod
    def plot_membership_functions(x, mfs, title="Membership Functions", save_path: str = None):
        """
        Vẽ các hàm thành viên (Fuzzification).
        x: array không gian mẫu (vd: np.linspace(0, 100, 100))
        mfs: list các mảng giá trị của hàm thành viên
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, mf in enumerate(mfs):
            color = colors[i % len(colors)]
            ax.plot(x, mf, label=f'MF {i+1}', color=color, linewidth=2)
            ax.fill_between(x, mf, alpha=0.2, color=color)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Input Value', fontsize=10)
        ax.set_ylabel('Degree of Membership (μ)', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close(fig)
        return fig

    @staticmethod
    def plot_anfis_architecture(save_path: str = None):
        """Vẽ sơ đồ mạng ANFIS 5 lớp cơ bản (2 Input, 2 Rule)."""
        fig, ax = plt.subplots(figsize=(12, 6))
        G = nx.DiGraph()

        # Định nghĩa tọa độ (x, y) cho từng node để tạo hình 5 lớp
        # Lớp 1: Input (x, y)
        G.add_node('x', pos=(0, 2)); G.add_node('y', pos=(0, -2))
        
        # Lớp 2: Fuzzification (A1, A2, B1, B2)
        G.add_node('A1', pos=(1, 3)); G.add_node('A2', pos=(1, 1))
        G.add_node('B1', pos=(1, -1)); G.add_node('B2', pos=(1, -3))
        
        # Lớp 3: Rule Firing (w1, w2)
        G.add_node('w1', pos=(2, 2)); G.add_node('w2', pos=(2, -2))
        
        # Lớp 4: Normalization (w1_bar, w2_bar)
        G.add_node('w1_bar', pos=(3, 2)); G.add_node('w2_bar', pos=(3, -2))
        
        # Lớp 5: Consequent & Lớp 6: Output
        G.add_node('f1', pos=(4, 2)); G.add_node('f2', pos=(4, -2))
        G.add_node('Output', pos=(5, 0))

        # Định nghĩa các cạnh nối (Edges)
        edges = [
            ('x', 'A1'), ('x', 'A2'), ('y', 'B1'), ('y', 'B2'), # Input -> Fuzzify
            ('A1', 'w1'), ('B1', 'w1'), ('A2', 'w2'), ('B2', 'w2'), # Fuzzify -> Rules
            ('w1', 'w1_bar'), ('w2', 'w2_bar'), ('w1', 'w2_bar'), ('w2', 'w1_bar'), # Rules -> Norm (Chéo nhau)
            ('w1_bar', 'f1'), ('w2_bar', 'f2'), # Norm -> Consequent
            ('f1', 'Output'), ('f2', 'Output') # Consequent -> Output
        ]
        G.add_edges_from(edges)

        # Lấy tọa độ để vẽ
        pos = nx.get_node_attributes(G, 'pos')
        
        # Phân biệt màu sắc cho các node ở các lớp khác nhau
        node_colors = ['#cccccc', '#cccccc', # Input
                       '#aec7e8', '#aec7e8', '#aec7e8', '#aec7e8', # Fuzzify
                       '#ffbb78', '#ffbb78', # Rules
                       '#98df8a', '#98df8a', # Norm
                       '#ff9896', '#ff9896', # Consequent
                       '#c5b0d5'] # Output

        # Vẽ đồ thị
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500, node_color=node_colors, edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='->', arrowsize=15, edge_color='gray')

        # Thêm text chú thích cho các lớp
        layers_labels = {0: "Lớp 1\nInput", 1: "Lớp 2\nFuzzification", 2: "Lớp 3\nRules", 
                         3: "Lớp 4\nNormalization", 4: "Lớp 5\nConsequent", 5: "Lớp 6\nOutput"}
        for x_coord, label in layers_labels.items():
            ax.text(x_coord, 4.5, label, horizontalalignment='center', fontweight='bold', fontsize=11)

        ax.set_title("Kiến trúc Mạng Lai ghép ANFIS (2 Inputs, 2 Rules)", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off') # Tắt trục tọa độ
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close(fig)
        return fig