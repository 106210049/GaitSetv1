import torch.utils.data as tordata  # Thư viện DataLoader và Sampler của PyTorch
import random                       # Dùng để chọn mẫu ngẫu nhiên (PID và index)



class TripletSampler(tordata.sampler.Sampler):  # Kế thừa từ Sampler của PyTorch
    def __init__(self, dataset, batch_size):
        self.dataset = dataset            # Đối tượng dataset (phải có label_set và index_dict)
        self.batch_size = batch_size      # batch_size = [P, M] → P người, mỗi người lấy M mẫu


    def __iter__(self):                   # Trình tạo dữ liệu cho mỗi batch
        while True:                       # Tạo vòng lặp vô hạn (DataLoader sẽ giới hạn vòng)
            sample_indices = list()      # Danh sách lưu chỉ số mẫu trong batch

            pid_list = random.sample(     # Chọn ngẫu nhiên P ID (người) từ tập nhãn
                list(self.dataset.label_set),
                self.batch_size[0])       # self.batch_size[0] là P (số lượng người trong batch)

            for pid in pid_list:          # Lặp qua từng người vừa chọn
                # Lấy tất cả chỉ số mẫu của người này từ index_dict (giá trị > 0 là hợp lệ)
                _index = self.dataset.index_dict.loc[pid, :, :].values  # Lấy toàn bộ chỉ số mẫu của PID
                _index = _index[_index > 0].flatten().tolist()          # Bỏ các giá trị <= 0 và flatten thành list

                # Chọn ngẫu nhiên M mẫu từ danh sách trên (cho phép trùng lặp)
                _index = random.choices(_index, k=self.batch_size[1])   # self.batch_size[1] là M

                sample_indices += _index   # Thêm các chỉ số này vào batch
            yield sample_indices           # Trả về batch tiếp theo (danh sách chỉ số P × M mẫu)


    def __len__(self):                    # Trả về tổng số mẫu trong dataset (gần đúng thôi)
        return self.dataset.data_size     # Được dùng để báo kích thước cho DataLoader

