import os                                 # Thư viện thao tác hệ điều hành
import os.path as osp                     # Thư viện thao tác đường dẫn
import numpy as np                        # Thư viện xử lý mảng số
from .data_set import DataSet             # Import lớp DataSet đã định nghĩa


# Hàm load_data: tải và chuẩn bị dữ liệu cho train/test
def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()                      # Danh sách đường dẫn tới thư mục chứa ảnh của từng chuỗi
    view = list()                         # Danh sách góc quay
    seq_type = list()                     # Danh sách loại chuỗi (nm-01, bg-01,...)
    label = list()                        # Danh sách ID người


        # Duyệt qua từng ID người (label)
    for _label in sorted(list(os.listdir(dataset_path))):
        if dataset == 'CASIA-B' and _label == '005':  # Loại bỏ ID 005 khỏi CASIA-B (thường bị lỗi)
            continue
        label_path = osp.join(dataset_path, _label)  # Đường dẫn đến thư mục người

        for _seq_type in sorted(list(os.listdir(label_path))):  # Duyệt loại chuỗi
            seq_type_path = osp.join(label_path, _seq_type)

            for _view in sorted(list(os.listdir(seq_type_path))):  # Duyệt góc quay
                _seq_dir = osp.join(seq_type_path, _view)          # Đường dẫn tới ảnh chuỗi cụ thể

                seqs = os.listdir(_seq_dir)                        # Danh sách ảnh trong chuỗi

                if len(seqs) > 0:                                  # Nếu có ảnh thì lưu lại
                    seq_dir.append([_seq_dir])                    # Lưu đường dẫn (dưới dạng list)
                    label.append(_label)                          # Ghi lại ID
                    seq_type.append(_seq_type)                    # Ghi loại chuỗi
                    view.append(_view)                            # Ghi góc quay


    # Tạo đường dẫn file lưu partition (train/test)
    pid_fname = osp.join('partition', '{}_{}_{}.npy'.format(
        dataset, pid_num, pid_shuffle))  # Ví dụ: CASIA-B_70_True.npy

        # Nếu chưa có file chia partition thì tạo mới
    if not osp.exists(pid_fname):
        pid_list = sorted(list(set(label)))       # Danh sách ID người không trùng
        if pid_shuffle:
            np.random.shuffle(pid_list)           # Nếu yêu cầu shuffle thì xáo trộn danh sách

        pid_dict = {                              # Tạo dict chứa ID cho train và test
            'train': np.array(pid_list[0:pid_num]),           # Lấy pid_num ID đầu tiên làm train
            'test': np.array(pid_list[pid_num:])              # Còn lại làm test
        }
        os.makedirs('partition', exist_ok=True)               # Tạo thư mục partition nếu chưa có
        np.save(pid_fname, pid_dict, allow_pickle=True)       # Lưu file partition dưới dạng .npy

    
    try:
        pid_dict = np.load(pid_fname, allow_pickle=True).item()   # Load file partition
        train_list = pid_dict['train'].tolist()                   # Lấy danh sách ID train
        test_list = pid_dict['test'].tolist()                     # Lấy danh sách ID test

    except (EOFError, IOError, KeyError):                         # Nếu file hỏng thì tạo lại
        pid_list = sorted(list(set(label)))
        if pid_shuffle:
            np.random.shuffle(pid_list)

        pid_dict = {
            'train': np.array(pid_list[0:pid_num]),
            'test': np.array(pid_list[pid_num:])
        }
        np.save(pid_fname, pid_dict, allow_pickle=True)
        train_list = pid_dict['train'].tolist()
        test_list = pid_dict['test'].tolist()


    # Tạo đối tượng DataSet cho train (lọc theo label thuộc train_list)
    train_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],     # Đường dẫn chuỗi ảnh train
        [label[i] for i, l in enumerate(label) if l in train_list],       # Nhãn train
        [seq_type[i] for i, l in enumerate(label) if l in train_list],    # Loại chuỗi train
        [view[i] for i, l in enumerate(label) if l in train_list],        # Góc quay train
        cache, resolution)                                                # Các tham số khác

    
    # Tạo đối tượng DataSet cho test (lọc theo label thuộc test_list)
    test_source = DataSet(
        [seq_dir[i] for i, l in enumerate(label) if l in test_list],
        [label[i] for i, l in enumerate(label) if l in test_list],
        [seq_type[i] for i, l in enumerate(label) if l in test_list],
        [view[i] for i, l in enumerate(label) if l in test_list],
        cache, resolution)


    return train_source, test_source   # Trả về hai tập dữ liệu train/test
