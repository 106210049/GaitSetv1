import torch.utils.data as tordata  # PyTorch DataLoader utilities
import numpy as np                 # Thư viện xử lý mảng số
import os.path as osp              # Thao tác đường dẫn (os.path)
import os                          # Thao tác hệ điều hành
import pickle                      # Đọc ghi file nhị phân (chưa dùng ở đây)
import cv2                         # Đọc ảnh
import xarray as xr                # Thư viện mảng dữ liệu kèm tọa độ (giống DataFrame cho tensor)



class DataSet(tordata.Dataset):  # Lớp dataset kế thừa từ PyTorch Dataset
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir                    # Danh sách đường dẫn thư mục ảnh
        self.view = view                          # Danh sách góc quay của các chuỗi ảnh
        self.seq_type = seq_type                  # Danh sách loại chuỗi (nm/bg/cl)
        self.label = label                        # Danh sách nhãn ID người
        self.cache = cache                        # Nếu True thì preload toàn bộ dữ liệu
        self.resolution = int(resolution)         # Kích thước ảnh sau resize (ví dụ 64)
        self.cut_padding = int(float(resolution)/64*10)  # Cắt padding hai bên ảnh (ví dụ: 10px nếu 64x64)
        self.data_size = len(self.label)          # Số mẫu
        self.data = [None] * self.data_size       # Lưu cache dữ liệu ảnh nếu cần
        self.frame_set = [None] * self.data_size  # Lưu set frame hợp lệ tương ứng
        self.label_set = set(self.label)          # Tập nhãn không trùng
        self.seq_type_set = set(self.seq_type)    # Tập seq_type không trùng
        self.view_set = set(self.view)            # Tập góc quay không trùng
        _ = np.zeros((len(self.label_set), len(self.seq_type_set), len(self.view_set))).astype('int')  # Khởi tạo mảng 3D index_dict
        _ -= 1                                     # Gán giá trị -1 mặc định cho index_dict
        self.index_dict = xr.DataArray(_, coords={'label': sorted(list(self.label_set)), 'seq_type': sorted(list(self.seq_type_set)), 'view': sorted(list(self.view_set))}, dims=['label', 'seq_type', 'view'])  # Biến thành DataArray có tọa độ
        for i in range(self.data_size):           # Gán chỉ số vào index_dict cho việc truy xuất nhanh
            _label = self.label[i]; _seq_type = self.seq_type[i]; _view = self.view[i]; self.index_dict.loc[_label, _seq_type, _view] = i


    def load_all_data(self):                      # Load toàn bộ dữ liệu ảnh (nếu cần preload)
        for i in range(self.data_size): self.load_data(i)

    def load_data(self, index):                   # Load 1 mẫu theo chỉ số
        return self.__getitem__(index)


    def __loader__(self, path):                   # Load ảnh từ thư mục → DataArray chuẩn hóa 0-1
        return self.img2xarray(path)[:, :, self.cut_padding:-self.cut_padding].astype('float32') / 255.0


    def __getitem__(self, index):  # Trả về 1 mẫu dữ liệu gồm: [list đặc trưng (DataArray), tập frame chung, view, seq_type, label]
        if not self.cache:  # Nếu không cache → load ảnh từ thư mục
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))  # Lấy frame chung của tất cả đặc trưng
        elif self.data[index] is None:  # Nếu cache nhưng chưa load
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set)); self.data[index] = data; self.frame_set[index] = frame_set
        else:  # Nếu cache và đã load rồi → dùng lại
            data = self.data[index]; frame_set = self.frame_set[index]
        return data, frame_set, self.view[index], self.seq_type[index], self.label[index]


    def img2xarray(self, flie_path):  # Đọc ảnh từ thư mục → chuyển thành xarray.DataArray
        imgs = sorted(list(os.listdir(flie_path)))  # Danh sách file ảnh
        frame_list = [np.reshape(cv2.imread(osp.join(flie_path, _img_path)), [self.resolution, self.resolution, -1])[:, :, 0] for _img_path in imgs if osp.isfile(osp.join(flie_path, _img_path))]  # Đọc và resize ảnh, lấy kênh đầu (đen trắng)
        num_list = list(range(len(frame_list)))     # Frame index
        data_dict = xr.DataArray(frame_list, coords={'frame': num_list}, dims=['frame', 'img_y', 'img_x'])  # Đóng gói thành DataArray
        return data_dict


    def __len__(self): return len(self.label)  # Trả về số lượng mẫu trong dataset
