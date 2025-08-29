# ==============================================================================
# PHẦN 1: IMPORT CÁC THƯ VIỆN
# ==============================================================================
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import ipywidgets as widgets
from IPython.display import display, clear_output

# ==============================================================================
# PHẦN 2: TẠO DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
n_samples = 100
sizes = np.random.randint(9, 15, n_samples)
weights = np.random.uniform(150, 500, n_samples)
surfaces = np.random.randint(1, 4, n_samples)
colors = np.random.randint(1, 4, n_samples)
X = np.column_stack((sizes, weights, surfaces, colors))

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
    final_class = max(1, min(3, final_class))
    y.append(final_class)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Hệ thống đã được huấn luyện và sẵn sàng để đánh giá. 🥭")
print("-" * 60)

# ==============================================================================
# PHẦN 3: GIAO DIỆN VÀ PHÂN TÍCH
# ==============================================================================
COLOR_PRIMARY = '#4CAF50'
COLOR_SECONDARY = '#FFC107'
COLOR_DANGER = '#DC3545'
COLOR_SUCCESS = '#28A745'
COLOR_INFO = '#17A2B8'

widget_style = {'description_width': '40%'}
widget_layout = widgets.Layout(width='48%')

title = widgets.HTML(f"<h3 style='color: {COLOR_PRIMARY};'><b style='font-size:24px;'>🥭 HỆ THỐNG ĐÁNH GIÁ CHẤT LƯỢNG XOÀI 🥭</b></h3>")
size_input = widgets.IntText(value=11, description='Kích thước (cm):', style=widget_style, layout=widget_layout)
weight_input = widgets.FloatText(value=300.0, description='Khối lượng (g):', style=widget_style, layout=widget_layout)
surface_dropdown = widgets.Dropdown(options=[('Không hư hỏng', 1), ('Khuyết điểm nhỏ', 2), ('Hư hỏng nặng', 3)], value=1, description='Tình trạng bề mặt:', style=widget_style, layout=widget_layout)
color_dropdown = widgets.Dropdown(options=[('Vàng (Chín tới)', 1), ('Xanh (Chưa chín)', 2), ('Cam (Chín quá)', 3)], value=1, description='Màu sắc (Độ chín):', style=widget_style, layout=widget_layout)
predict_button = widgets.Button(description='Thực Hiện Phân Tích', button_style='success', icon='cogs', layout=widgets.Layout(width='98%', margin='20px 0 0 0'))
output_area = widgets.Output(layout=widgets.Layout(margin='20px 0 0 0', padding='15px', border=f'1px solid {COLOR_PRIMARY}', width='98%', background_color='#F8F9FA'))

def on_button_clicked(b):
    size, weight, surface, color = size_input.value, weight_input.value, surface_dropdown.value, color_dropdown.value
    new_mango_scaled = scaler.transform(np.array([[size, weight, surface, color]]))
    predicted_class = model.predict(new_mango_scaled)[0]

    # --- PHÂN LOẠI MỞ RỘNG ---
    if surface == 3:
        predicted_class = 3
    elif color == 3:
        predicted_class = 2 if surface != 3 else 3
    elif size >= 13 and surface <=2 and color != 3:
        predicted_class = 1
    else:
        predicted_class = 2

    # --- PHÂN TÍCH THỰC TẾ VỚI NHIỀU MẪU NGẪU NHIÊN ---
    analysis_options = []
    recommendation_options = []
    storage_options = []
    result_title_color = COLOR_PRIMARY

    if predicted_class == 1:
        result_title_color = COLOR_SUCCESS
        if surface ==1 and color ==1:
            analysis_options = [
                "Xoài cực ngon, vỏ đẹp, vừa chín tới, kích thước lớn.",
                "Xoài tươi, ăn giòn, vỏ đẹp, rất thích hợp biếu tặng.",
                "Quả ngon nhất lứa, cân nặng vừa phải, vỏ mịn."
            ]
            recommendation_options = [
                "Ưu tiên lựa chọn để ăn sống hoặc biếu tặng.",
                "Chọn làm quà hoặc ăn ngay, vị ngọt tự nhiên.",
                "Đặt lên mâm cúng hay dùng để làm món tráng miệng."
            ]
            storage_options = [
                "Bảo quản ở nhiệt độ phòng 2-3 ngày hoặc tủ lạnh 5-7 ngày.",
                "Để nơi thoáng mát, tránh ánh nắng trực tiếp.",
                "Bảo quản lạnh nếu chưa dùng, dùng trong 1 tuần."
            ]
        elif surface ==2:
            analysis_options = [
                "Xoài ngon, chỉ có vài vết xước nhỏ.",
                "Quả hơi khuyết trên vỏ nhưng thịt vẫn ngọt.",
                "Vỏ không hoàn hảo, nhưng chất lượng vẫn cao."
            ]
            recommendation_options = [
                "Chất lượng cao, vẫn ưu tiên lựa chọn.",
                "Ăn ngon, dùng để chế biến hay biếu đều được.",
                "Dùng ngay hoặc để vài ngày đều ổn."
            ]
            storage_options = [
                "Nhiệt độ phòng 2-3 ngày, hoặc bảo quản lạnh 5-6 ngày.",
                "Đặt nơi thoáng mát, tránh ánh nắng trực tiếp.",
                "Bảo quản lạnh nếu muốn dùng lâu hơn."
            ]
        else:
            analysis_options = ["Xoài ngon nhưng vỏ hơi xước, vẫn đáng ăn."]
            recommendation_options = ["Loại 1, dùng ăn ngon hoặc chế biến."]
            storage_options = ["Bảo quản phòng 2-3 ngày hoặc tủ lạnh 5-6 ngày."]

    elif predicted_class == 2:
        result_title_color = COLOR_SECONDARY
        if color ==3:
            analysis_options = [
                "Xoài chín quá, vỏ mềm, ăn vẫn được nhưng không còn loại 1.",
                "Quả chín nặng, vỏ vàng cam, thịt hơi mềm.",
                "Chín quá, vị vẫn ngọt nhưng ăn sống hơi nhũn."
            ]
            recommendation_options = [
                "Loại 2, dùng ngay hoặc làm sinh tố, mứt.",
                "Ăn ngay để tránh nát, chế biến thành món tráng miệng.",
                "Dùng cho sinh tố, salad, hoặc nấu ăn."
            ]
        else:
            analysis_options = [
                "Xoài trung bình, vỏ có khuyết điểm, chưa chín đều.",
                "Quả vừa, ăn vẫn ngon nhưng không xuất sắc.",
                "Kích thước trung bình, vài vết xước nhỏ."
            ]
            recommendation_options = [
                "Loại 2, cân nhắc khi mua, ăn sau vài ngày.",
                "Ăn ngon nhưng không xuất sắc, nên ăn từ từ.",
                "Dùng chế biến hoặc ăn trực tiếp sau vài ngày ủ."
            ]
        storage_options = [
            "Bảo quản nhiệt độ phòng 2-4 ngày, hoặc lạnh ngắn hạn.",
            "Để nơi thoáng mát, tránh ánh nắng trực tiếp.",
            "Bảo quản lạnh nếu chưa ăn ngay, dùng trong 3-4 ngày."
        ]

    else:  # Loại 3
        result_title_color = COLOR_INFO
        if surface ==3:
            analysis_options = [
                "Xoài hư hỏng nặng, vỏ dập, mềm, ăn sống không ngon.",
                "Vỏ nát, quả mềm, chỉ dùng chế biến.",
                "Hư hỏng nhiều, không thích hợp ăn trực tiếp."
            ]
            recommendation_options = [
                "Loại 3, chỉ dùng chế biến hoặc bỏ.",
                "Dùng để làm mứt hoặc sinh tố ngay.",
                "Không ăn sống, chế biến ngay nếu muốn sử dụng."
            ]
        else:
            analysis_options = [
                "Xoài chín quá, vỏ vàng cam đậm, mềm, dễ nát.",
                "Quả quá chín, ăn ngay hoặc chế biến.",
                "Chín nặng, thịt mềm, vị ngọt nhưng không còn tươi."
            ]
            recommendation_options = [
                "Loại 3, dùng ngay để ăn hoặc chế biến.",
                "Ăn ngay hoặc làm sinh tố, mứt.",
                "Không lưu trữ lâu, dùng trong ngày."
            ]
        storage_options = [
            "Bảo quản lạnh tối đa 1 ngày nếu chưa dùng.",
            "Dùng ngay để tránh nát.",
            "Không để ngoài trời, bảo quản ngắn hạn."
        ]

    # Chọn ngẫu nhiên trong từng nhóm
    analysis_report = np.random.choice(analysis_options)
    recommendation = np.random.choice(recommendation_options)
    storage_guide = np.random.choice(storage_options)

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
                <p><strong>1. Phân Tích Thực Tế:</strong></p>
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

row_layout_with_margin = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between', width='100%', margin='20px 0 0 0')
row1 = widgets.Box([size_input, surface_dropdown], layout=row_layout_with_margin)
row2 = widgets.Box([weight_input, color_dropdown], layout=widgets.Layout(display='flex', flex_flow='row', justify_content='space-between', width='100%', margin='10px 0 0 0'))
app_layout = widgets.VBox([title, row1, row2, predict_button, output_area], layout=widgets.Layout(align_items='center', width='700px', border=f'2px solid {COLOR_PRIMARY}', padding='20px', background_color='#E8F5E9'))

display(app_layout)
