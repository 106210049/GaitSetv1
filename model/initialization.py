# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15

import os                         # Thư viện thao tác hệ điều hành (thay đổi thư mục, biến môi trường)
from copy import deepcopy         # Dùng để sao chép dict (tránh sửa gốc)
import numpy as np                # Thư viện xử lý mảng số

from .utils import load_data      # Hàm load dữ liệu từ thư mục
from .model import Model          # Import lớp mô hình chính (huấn luyện, test,...)



def initialize_data(config, train=False, test=False):
    print("Initializing data source...")

    # Gọi hàm load_data từ file utils.py để lấy đối tượng train_source và test_source
    # cache=True nếu là train hoặc test để preload dữ liệu
    train_source, test_source = load_data(**config['data'], cache=(train or test))

    # Nếu là train → load toàn bộ dữ liệu train vào RAM (tăng tốc training)
    if train:
        print("Loading training data...")
        train_source.load_all_data()

    # Nếu là test → load toàn bộ dữ liệu test vào RAM
    if test:
        print("Loading test data...")
        test_source.load_all_data()

    print("Data initialization complete.")  # In thông báo khi hoàn tất
    return train_source, test_source        # Trả về dataset cho train và test



def initialize_model(config, train_source, test_source):
    print("Initializing model...")

    data_config = config['data']         # Lấy config phần dữ liệu
    model_config = config['model']       # Lấy config phần mô hình
    model_param = deepcopy(model_config) # Sao chép config mô hình để chỉnh sửa

    # Gán thêm thông tin dữ liệu vào model_param
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']  # Số lượng ID người dùng cho training

    # Tính batch_size thực tế (P * M)
    batch_size = int(np.prod(model_config['batch_size']))

    # Tạo tên file lưu model dựa trên các thông số cấu hình → để dễ phân biệt các mô hình đã train
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],      # Tên model (ví dụ: GaitSet)
        data_config['dataset'],          # Tên dataset (CASIA-B,...)
        data_config['pid_num'],          # Số người train
        data_config['pid_shuffle'],      # Có xáo trộn ID không
        model_config['hidden_dim'],      # Kích thước đầu ra encoder
        model_config['margin'],          # Margin của triplet loss
        batch_size,                      # Kích thước batch thực tế
        model_config['hard_or_full_trip'], # Loại triplet loss
        model_config['frame_num'],       # Số lượng frame sử dụng
    ]))

    m = Model(**model_param)  # Khởi tạo mô hình bằng các tham số đã gom lại
    print("Model initialization complete.")
    return m, model_param['save_name']  # Trả về model và tên để lưu checkpoint



def initialization(config, train=False, test=False):
    print("Initialzing...")

    WORK_PATH = config['WORK_PATH']                     # Lấy thư mục làm việc
    os.chdir(WORK_PATH)                                # Đổi thư mục hiện tại sang WORK_PATH
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]  # Thiết lập GPU nào sẽ được sử dụng

    # Gọi hàm khởi tạo dữ liệu
    train_source, test_source = initialize_data(config, train, test)

    # Gọi hàm khởi tạo mô hình và trả về kết quả cuối cùng
    return initialize_model(config, train_source, test_source)
