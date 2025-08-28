# ==============================================================================
# PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==============================================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==============================================================================
# PHẦN 2: TẠO DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH (Giữ nguyên)
# ==============================================================================

# --- Tạo dữ liệu mẫu ---
n_samples = 100
sizes = np.random.randint(9, 15, n_samples)
weights = np.random.uniform(150, 500, n_samples)
surfaces = np.random.randint(1, 4, n_samples)
colors = np.random.randint(1, 4, n_samples)
X = np.column_stack((sizes, weights, surfaces, colors))

# --- Hàm phân loại và dán nhãn ---
def get_size_class(cm):
    if cm >= 13: return 1
    elif cm >= 11: return 2
    else: return 3
def get_weight_class(g):
    if g >= 350: return 1
    elif g >= 250: return 2
    else: return 3
y = []
for features in X:
    c1 = get_size_class(features[0])
    c2 = get_weight_class(features[1])
    c3 = features[2]
    c4 = features[3]
    final_class = int(np.round(np.mean([c1, c2, c3, c4])))
    if final_class < 1: final_class = 1
    if final_class > 3: final_class = 3
    y.append(final_class)
y = np.array(y)

# --- Huấn luyện mô hình ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Hệ thống đã được huấn luyện và sẵn sàng để đánh giá. 🥭")
print("-" * 60)


# ==============================================================================
# PHẦN 3: GIAO DIỆN KẾT HỢP PHÂN LOẠI VÀ PHÂN TÍCH
# ==============================================================================

# --- Định nghĩa màu sắc chủ đạo ---
COLOR_PRIMARY = '#4CAF50' # Xanh lá cây
COLOR_SECONDARY = '#FFC107' # Vàng xoài
COLOR_DANGER = '#DC3545' # Đỏ cảnh báo
COLOR_SUCCESS = '#28A745' # Xanh lá thành công

# --- Tạo các thành phần (widgets) ---
widget_style = {'description_width': '40%'}
widget_layout = widgets.Layout(width='48%')

title = widgets.HTML(f"<h3 style='color: {COLOR_PRIMARY};'><b style='font-size:24px;'>🥭 HỆ THỐNG ĐÁNH GIÁ CHẤT LƯỢNG XOÀI 🥭</b></h3>")
size_input = widgets.IntText(value=11, description='Kích thước (cm):', style=widget_style, layout=widget_layout)
weight_input = widgets.FloatText(value=300.0, description='Khối lượng (g):', style=widget_style, layout=widget_layout)
surface_dropdown = widgets.Dropdown(options=[('Không hư hỏng', 1), ('Khuyết điểm nhỏ', 2), ('Hư hỏng nặng', 3)], value=1, description='Tình trạng bề mặt:', style=widget_style, layout=widget_layout)
color_dropdown = widgets.Dropdown(options=[('Vàng (Chín tới)', 1), ('Xanh (Chưa chín)', 2), ('Cam (Chín quá)', 3)], value=1, description='Màu sắc (Độ chín):', style=widget_style, layout=widget_layout)
predict_button = widgets.Button(description='Thực Hiện Phân Tích', button_style='success', icon='cogs', layout=widgets.Layout(width='98%', margin='20px 0 0 0'))
output_area = widgets.Output(layout=widgets.Layout(margin='20px 0 0 0', padding='15px', border=f'1px solid {COLOR_PRIMARY}', width='98%', background_color='#F8F9FA'))


# --- Định nghĩa hàm xử lý sự kiện (ĐÃ CẬP NHẬT) ---
def on_button_clicked(b):
    size, weight, surface, color = size_input.value, weight_input.value, surface_dropdown.value, color_dropdown.value
    
    # --- BƯỚC 1: DÙNG MÔ HÌNH ĐỂ PHÂN LOẠI ---
    new_mango_scaled = scaler.transform(np.array([[size, weight, surface, color]]))
    predicted_class = model.predict(new_mango_scaled)[0]

    # --- BƯỚC 2: PHÂN TÍCH CHUYÊN SÂU ĐỂ TƯ VẤN ---
    analysis_report = ""
    recommendation = ""
    storage_guide = ""
    result_title_color = COLOR_PRIMARY

    if surface == 3:
        analysis_report = "<b>Phân tích:</b> Bề mặt vỏ có dấu hiệu hư hỏng cơ học nghiêm trọng..."
        recommendation = "<b>Kết luận:</b> Sản phẩm không đạt tiêu chuẩn. <b><u>Khuyến nghị: Không lựa chọn.</u></b>"
        storage_guide = "<b>Bảo quản:</b> Cần được loại bỏ..."
        result_title_color = COLOR_DANGER
    elif color == 3:
        analysis_report = "<b>Phân tích:</b> Màu cam đậm là chỉ báo sản phẩm đã bước vào giai đoạn chín quá..."
        recommendation = "<b>Kết luận:</b> Chất lượng đang suy giảm nhanh. <b><u>Khuyến nghị: Chỉ lựa chọn để sử dụng ngay.</u></b>"
        storage_guide = "<b>Bảo quản:</b> Yêu cầu bảo quản lạnh và sử dụng trong vòng 24 giờ."
        result_title_color = COLOR_SECONDARY
    elif color == 1:
        if surface == 1:
            analysis_report = "<b>Phân tích:</b> Sản phẩm đạt các chỉ số tối ưu..."
            recommendation = "<b>Kết luận:</b> Sản phẩm đạt chất lượng cao. <b><u>Khuyến nghị: Ưu tiên lựa chọn hàng đầu.</u></b>"
        else:
            analysis_report = "<b>Phân tích:</b> Bề mặt vỏ có khuyết điểm nhỏ không ảnh hưởng đáng kể..."
            recommendation = "<b>Kết luận:</b> Chất lượng ở mức chấp nhận được. <b><u>Khuyến nghị: Có thể lựa chọn.</u></b>"
        storage_guide = "<b>Bảo quản:</b> Tối ưu ở nhiệt độ phòng (2-3 ngày) hoặc bảo quản lạnh (5-7 ngày)."
        result_title_color = COLOR_SUCCESS
    elif color == 2:
        analysis_report = "<b>Phân tích:</b> Vỏ màu xanh là chỉ báo xoài chưa đạt độ chín thương phẩm..."
        if surface == 1:
            recommendation = "<b>Kết luận:</b> Sản phẩm có tiềm năng tốt sau ủ chín. <b><u>Khuyến nghị: Lựa chọn để ủ chín hoặc ăn chua.</u></b>"
        else:
            recommendation = "<b>Kết luận:</b> Chất lượng tiềm năng ở mức trung bình. <b><u>Khuyến nghị: Cân nhắc kỹ lưỡng.</u></b>"
        storage_guide = "<b>Bảo quản để ủ chín:</b> Giữ ở nhiệt độ phòng, nơi thoáng mát. Thời gian ủ dự kiến: 3-5 ngày."
        result_title_color = '#007BFF'

    # Hiển thị kết quả
    with output_area:
        clear_output(wait=True)
        html_output = f"""
        <div>
            <h4 style='text-align: center; color: {result_title_color}; font-size: 20px;'>BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG SẢN PHẨM</h4>
            
            <div style='background-color: {result_title_color}; color: white; padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center;'>
                <span style='font-size: 18px;'>PHÂN LOẠI THEO MÔ HÌNH: <b>🥭 XOÀI LOẠI {predicted_class} 🥭</b></span>
            </div>
            
            <hr style='border-top: 1px solid {COLOR_PRIMARY};'>
            <div style='text-align: left;'>
                <p><strong>1. Phân Tích Chuyên Sâu:</strong></p>
                <p>{analysis_report}</p>
                <br>
                <p><strong>2. Kết Luận & Khuyến Nghị:</strong></p>
                <p style='color: {result_title_color};'>{recommendation}</p>
                <br>
                <p><strong>3. Hướng Dẫn Bảo Quản & Sử Dụng:</strong></p>
                <p>{storage_guide}</p>
            </div>
        </div>
        """
        display(widgets.HTML(value=html_output))

predict_button.on_click(on_button_clicked)

# --- Sắp xếp layout bằng Flexbox ---
row_layout_with_margin = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between', width='100%', margin='20px 0 0 0')
row1 = widgets.Box([size_input, surface_dropdown], layout=row_layout_with_margin)
row2 = widgets.Box([weight_input, color_dropdown], layout=widgets.Layout(display='flex', flex_flow='row', justify_content='space-between', width='100%', margin='10px 0 0 0'))
app_layout = widgets.VBox([title, row1, row2, predict_button, output_area], layout=widgets.Layout(align_items='center', width='700px', border=f'2px solid {COLOR_PRIMARY}', padding='20px', background_color='#E8F5E9'))

# --- Hiển thị giao diện ---
display(app_layout)