# -*- coding: utf-8 -*-
# @Author  : Abner
# @Time    : 2018/12/19
# ******************** Pretreatment ********************
'''
Đoạn code thực hiện tiền xử lý ảnh - cụ thể là xử lý các ảnh silhouette (dạng đen trắng) 
của người, chuẩn hóa kích thước, căn giữa, và lưu lại kết quả đã xử lý. 
Các bước xử lý gồm:
B1: Đọc ảnh silhouette từ thư mục dữ liệu đầu vào (input_path).
B2: Cắt gọn và căn chỉnh hình ảnh.
B3: Chuẩn hóa kích thước về chuẩn (64 x 64).
B4: Chuẩn hóa kích thước về chuẩn (64 x 64).
B5: Ghi log trong quá trình xử lý.
'''
# *******************************************************

# ******************** Thêm các thư viện xử lý ********************
import os
from scipy import misc as scisc
import cv2  # Đọc file và xử lý ảnh
import numpy as np
from warnings import warn
from time import sleep
import argparse
import imageio #ghi ảnh ra file.
from multiprocessing import Pool    #xử lý song song để tăng tốc.
from multiprocessing import TimeoutError as MP_TimeoutError
# ****************************************************************

# ******************** Khai báo biến và hằng số ********************
START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"
# ******************************************************************

# ******************** Hàm xử lý boolean trong argparse ********************
# Chức năng: Chuyển đối số kiểu chuỗi "TRUE" hoặc "FALSE" thành kiểu bool.
def boolean_string(s):
    #Chuyển chuỗi s sang chữ in hoa (s.upper()).
    #Kiểm tra nếu không phải là "TRUE" hoặc "FALSE" → thì không hợp lệ.
    if s.upper() not in {'FALSE', 'TRUE'}:
        # Nếu chuỗi không hợp lệ → báo lỗi ValueError với thông báo rõ ràng.
        raise ValueError('Not a valid boolean string')
    # Nếu chuỗi hợp lệ, kiểm tra nó có bằng "TRUE" không.
    # Nếu có → trả về True, ngược lại trả về False.
    return s.upper() == 'TRUE'
# *************************************************************************

# ******************** Đọc đối số dòng lệnh ********************
#Tạo một đối tượng ArgumentParser để định nghĩa và xử lý các thông số (arguments) truyền từ dòng lệnh.
#description='Test' sẽ được hiển thị khi bạn chạy lệnh python script.py --help.
parser = argparse.ArgumentParser(description='Test')

#********** Các tham số được định nghĩa **********
# --input_path
# Mục đích: chỉ định thư mục chứa dữ liệu ảnh gốc.
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
# --output_path
# Mục đích: nơi lưu dữ liệu đầu ra (ảnh đã xử lý).
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')
# --log file
# Mục đích: chỉ định file để ghi log xử lý.
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
# --log
#Ý nghĩa:
# Nếu là True: lưu tất cả log (bao gồm START, FINISH).
# Nếu là False: chỉ lưu log loại WARNING, FAIL.
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. '
                         'Otherwise, only warnings and errors will be saved.'
                         'Default: False')
# --worker_num:
# Mục đích: Dùng để xác định số tiến trình xử lý song song.
parser.add_argument('--worker_num', default=1, type=int,
                    help='How many subprocesses to use for data pretreatment. '
                         'Default: 1')
#************************************************


# ********** Parse tất cả tham số **********
# Dòng này sẽ lấy tất cả các tham số mà người dùng truyền từ dòng lệnh và lưu trong đối tượng opt.
opt = parser.parse_args()
#************************************************
# *************************************************************************


INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num

# T_H và T_W: kích thước chuẩn (height và width) của ảnh đầu ra.
T_H = 64
T_W = 64


# ************************************** Các hàm ghi Log file **************************************
# chức năng: Tạo chuỗi log định dạng. Ghi log ra file hoặc in ra màn hình nếu cần.

#********** Hàm log2str **********
# Mục đích: Tạo ra một chuỗi định dạng log từ các tham số đầu vào để dễ ghi log hoặc in ra.
def log2str(pid, comment, logs):    # Hàm nhận 3 tham số: pid, comment, logs
    str_log = ''    # khởi tạo chuỗi log rỗng
    '''Nếu logs chỉ là 1 chuỗi duy nhất thì chuyển nó thành một list chứa 
    1 phần tử, để xử lý thống nhất trong vòng lặp bên dưới.'''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        # Với mỗi log trong danh sách logs, định dạng chuỗi log kiểu: JOB 12 : --WARNING-- seq:001, frame:0002.png, no data, 0.
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    # Trả lại chuỗi log hoàn chỉnh.
    return str_log
#*********************************

#********** Hàm log_print **********
# Chức năng: Ghi log ra file hoặc màn hình console, tùy theo loại log.
def log_print(pid, comment, logs):
    # Gọi lại hàm log2str để tạo chuỗi log định dạng.
    str_log = log2str(pid, comment, logs)
    # Ghi gile nếu là cảnh báo/ lỗi, Gọi lại hàm trên để tạo chuỗi log định dạng.
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    # In ra console nếu là START hoặc FINISH (nhưng giới hạn):
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    #In log ra console (không thêm dòng trống).
    print(str_log, end='')
#*********************************
# **********************************************************************************


# **************************** Hàm cắt ảnh ************************************
'''
Các bước thực hiện:
B1	Bỏ qua ảnh quá tối (nhiễu, sai)
B2	Xác định vùng chứa hình người
B3	Cắt biên trên - dưới của người
B4	Resize theo tỉ lệ chiều cao → chuẩn T_H
B5	Căn giữa người theo trục ngang (X)
B6	Cắt ảnh 64x64 theo vị trí trung tâm
B7	Trả về ảnh kết quả (uint8)
'''
def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    # ************ Bỏ qua ảnh quá tối (Dưới ngưỡng pixel trắng) ************
    if img.sum() <= 10000: 
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # *********************************************************************

    # Get the top and bottom point
    y = img.sum(axis=1)
    #************ Xác định biên trên và dưới của hình người ************
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # ******************************************************************

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    #************ Resize ảnh theo tỉ lệ chiều cao ************
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # ********************************************************
   
    # Get the median of x axis and regard it as the x center of the person.
    #************ Tìm trung điểm x (căn giữa hình) ************
    sum_point = img.sum() #Tổng toàn ảnh → sum_point.
    sum_column = img.sum(axis=0).cumsum() # sum_column: cộng dồn theo cột (từ trái qua phải).
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:   #Tìm cột ở giữa người → cột mà tổng bên trái = 50% tổng ảnh.
            x_center = i # Đây là tâm X của người → để crop giữa chính xác.
            break
    # ********************************************************
    
    #************ Xử lý lỗi không tìm được trung tâm ************
    # Nếu không tìm được trung tâm -> log lỗi -> bỏ ảnh
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # ********************************************************

    #************ Cắt ảnh theo trung tâm (về kích thước 64x64) ************
    #Tính vị trí cắt ảnh sao cho tâm người nằm giữa, chiều rộng 64.
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    # Nếu cắt vượt trái/phải ảnh → đệm cột đen (0) 2 bên ảnh để đủ crop.
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W)) # np.zeros: tạo ảnh đen để ghép vào.
        img = np.concatenate([_, img, _], axis=1)   #np.concatenate: nối mảng theo chiều ngang.
    img = img[:, left:right]    #Cuối cùng, cắt ảnh chính xác từ left đến right → ảnh chuẩn 64 x 64.
    return img.astype('uint8')  #Chuyển kiểu ảnh sang uint8 (chuẩn ảnh: 0–255 pixel).
    # ********************************************************
# ***********************************************************************


# **************************** Hàm xử lý 1 chuỗi ảnh ****************************
'''
Mục đích: 
- Xử lý toàn bộ ảnh trong một thư mục chứa ảnh silhouette.
- Gọi cut_img() để xử lý từng ảnh.
- Lưu ảnh đã xử lý ra thư mục đích.
- Ghi log quá trình làm.
'''
def cut_pickle(seq_info, pid):
    #B ắt đầu xử lý sequence
    #seq_info là list gồm 3 phần: [id, seq_type, view], ví dụ: ['001', 'nm', '000']
    # Gộp lại thành chuỗi seq_name = "001-nm-000" để ghi log.
    seq_name = '-'.join(seq_info)   
    
    # Ghi log bắt đầu xử lý sequence với mã tiến trình pid:
    log_print(pid, START, seq_name)
    
    # Xác định đường dẫn vào và ra
    seq_path = os.path.join(INPUT_PATH, *seq_info)  #seq_path: thư mục chứa ảnh gốc cần xử lý. Ví dụ: "data/001/nm/000"
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)  #out_dir: thư mục lưu ảnh sau khi cắt. Ví dụ: "processed/001/nm/000"
    
    # Đọc danh sách ảnh trong thư mục
    frame_list = os.listdir(seq_path)   #frame_list: danh sách tên file ảnh, ví dụ: ["0001.png", "0002.png", ...]
    frame_list.sort()   #sort() để đảm bảo xử lý theo thứ tự thời gian.
    
    count_frame = 0 #Biến đếm count_frame: số ảnh xử lý thành công.
    # Vòng lặp xử lý từng ảnh
    for _frame_name in frame_list:
        # Đọc ảnh gốc
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]   # Đọc ảnh với cv2.imread() chỉ lấy kênh màu đầu tiên
        
        # Tiền xử lý ảnh
        img = cut_img(img, seq_info, _frame_name, pid)  #Gọi hàm cut_img() để xử lý ảnh (crop, resize, căn giữa,...)
        
        #Lưu ảnh hợp lệ: 
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name) #Tạo đường dẫn lưu ảnh kết quả  
            imageio.imwrite(save_path, img) #Dùng imageio.imwrite() để lưu ảnh
            count_frame += 1 #Tăng count_frame lên

    # Warn if the sequence contains less than 5 frames
    if count_frame < 5: #Nếu sequence có quá ít ảnh hợp lệ (< 5) → ghi cảnh báo.
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)
    #Ghi log hoàn tất
    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))

# *******************************************************************************

#************************ Đoạn chương trình chính ************************
'''
Các bước thực hiện chính:
B1 Tạo pool: Dùng nhiều tiến trình song song để xử lý dữ liệu
B2 Duyệt thư mục ảnh
B3 Gọi hàm cut_pickel
B4 Theo dõi kết quả
B5 Xử lý lỗi
'''
# khởi tạo Pool và biến phụ trợ
pool = Pool(WORKERS)    # Tạo một pool tiến trình song song với số lượng WORKERS được chỉ định (dùng multiprocessing)
results = list()        # Danh sách lưu các đối tượng AsyncResult (kết quả xử lý từ pool.apply_async)
pid = 0                 # Biến đếm số job đã gửi vào pool, đồng thời dùng làm ID ghi log cho mỗi job

# In ra thông tin cấu hình trước khi bắt đầu xử lý
print('Pretreatment Start.\n'
      'Input path: %s\n'
      'Output path: %s\n'
      'Log file: %s\n'
      'Worker num: %d' % (
          INPUT_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

# Lấy danh sách tất cả các thư mục ID trong thư mục INPUT_PATH
id_list = os.listdir(INPUT_PATH)
id_list.sort()  # Sắp xếp theo thứ tự tăng dần để xử lý ổn định

# Walk the input path - Duyệt qua toàn bộ cấu trúc thư mục dữ liệu
for _id in id_list:
    seq_type = os.listdir(os.path.join(INPUT_PATH, _id))  # Lấy danh sách các loại sequence (vd: nm, bg, cl)
    seq_type.sort()  # Sắp xếp
    for _seq_type in seq_type:
        view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))  # Lấy danh sách các góc quay (vd: 000, 018, ...)
        view.sort()  # Sắp xếp
        for _view in view:
            seq_info = [_id, _seq_type, _view]  # Tập hợp thông tin định danh 1 chuỗi ảnh silhouette
            out_dir = os.path.join(OUTPUT_PATH, *seq_info)  # Tạo đường dẫn thư mục đầu ra tương ứng
            os.makedirs(out_dir)  # Tạo thư mục lưu ảnh đã xử lý (nếu chưa tồn tại)

            # Thêm một job xử lý chuỗi ảnh vào pool
            results.append(
                pool.apply_async(             # Gửi task xử lý vào pool dưới dạng không đồng bộ
                    cut_pickle,              # Hàm xử lý 1 sequence ảnh
                    args=(seq_info, pid)))   # Tham số truyền vào: thông tin chuỗi ảnh và ID tiến trình
            sleep(0.02)  # Tạm nghỉ 20ms giữa các job để tránh tạo quá nhiều tiến trình một lúc
            pid += 1     # Tăng ID tiến trình lên

# Đóng pool lại để không nhận thêm job mới
pool.close()

# Theo dõi các tiến trình xử lý cho đến khi tất cả hoàn tất
unfinish = 1
while unfinish > 0:
    unfinish = 0  # Giả định ban đầu là không còn tiến trình nào chưa hoàn tất
    for i, res in enumerate(results):  # Duyệt qua từng tiến trình
        try:
            res.get(timeout=0.1)  # Thử lấy kết quả với timeout ngắn (0.1s) để tránh block
        except Exception as e:
            if type(e) == MP_TimeoutError:
                unfinish += 1      # Nếu tiến trình chưa hoàn thành → tăng unfinish để lặp lại vòng while
                continue
            else:
                # Nếu xảy ra lỗi khác (ngoài timeout) → in lỗi và dừng chương trình
                print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                      i, type(e))
                raise e  # Ném lại lỗi để chương trình dừng hoàn toàn

# Đợi tất cả các tiến trình xử lý xong trước khi kết thúc chương trình
pool.join()

# *******************************************************************************
