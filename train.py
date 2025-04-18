from model.initialization import initialization  # Import hàm khởi tạo mô hình từ module model
from config import conf                         # Import cấu hình mô hình từ file config.py
import argparse                                 # Thư viện dùng để lấy tham số từ dòng lệnh

# Hàm chuyển chuỗi thành boolean (dùng cho argparse)
def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')  # Nếu không hợp lệ thì báo lỗi
    return s.upper() == 'TRUE'  # Trả về True nếu là 'TRUE', False nếu là 'FALSE'

# Khởi tạo bộ phân tích tham số dòng lệnh
parser = argparse.ArgumentParser(description='Train')

# Thêm đối số --cache: xác định có load toàn bộ dữ liệu train vào bộ nhớ hay không
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')

# Lấy các đối số đã phân tích được
opt = parser.parse_args()

# Khởi tạo mô hình (m), truyền vào config và cờ train=opt.cache để chỉ định chế độ load dữ liệu
m = initialization(conf, train=opt.cache)[0]

# Load checkpoint đã train tới vòng lặp thứ 7300 (tải trọng số đã lưu trước đó)
m.load(7300) 

# Thiết lập lại vòng lặp khôi phục để tiếp tục train từ checkpoint 7300
m.restore_iter = 7300  

# In ra thông báo bắt đầu training
print("Training START")

# Gọi hàm fit() để bắt đầu huấn luyện mô hình
m.fit()

# In ra thông báo kết thúc huấn luyện
print("Training COMPLETE")
