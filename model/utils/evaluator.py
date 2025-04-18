import torch                              # Thư viện tensor và deep learning
import torch.nn.functional as F           # Dùng hàm relu, softmax,...
import numpy as np                        # Xử lý mảng numpy


# Hàm tính ma trận khoảng cách Euclidean giữa 2 tập đặc trưng
def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()        # Chuyển x từ numpy array sang tensor và đẩy lên GPU
    y = torch.from_numpy(y).cuda()        # Chuyển y từ numpy array sang tensor và đẩy lên GPU

    # Công thức: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + \
           torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) - \
           2 * torch.matmul(x, y.transpose(0, 1))  # Ma trận khoảng cách [n_probe, n_gallery]

    dist = torch.sqrt(F.relu(dist))       # Lấy căn bậc 2 (tránh âm bằng relu)
    return dist                           # Trả về ma trận khoảng cách Euclidean



# Hàm đánh giá độ chính xác theo chuẩn CASIA-B hoặc OUMVLP
def evaluation(data, config):
    dataset = config['dataset'].split('-')[0]   # Lấy tên dataset (VD: "CASIA" từ "CASIA-B")

    feature, view, seq_type, label = data       # Giải nén đầu vào: đặc trưng, góc quay, loại chuỗi, nhãn
    label = np.array(label)                     # Chuyển label sang numpy array để so sánh

    view_list = list(set(view))                 # Lấy danh sách các góc quay duy nhất
    view_list.sort()                            # Sắp xếp tăng dần
    view_num = len(view_list)                   # Số lượng góc quay trong tập dữ liệu
    sample_num = len(feature)                   # Tổng số mẫu


    # Định nghĩa các chuỗi cho probe và gallery tùy theo dataset
    probe_seq_dict = {
        'CASIA': [['nm-05', 'nm-06'],           # Bình thường (Normal)
                  ['bg-01', 'bg-02'],           # Mang balo (Bag)
                  ['cl-01', 'cl-02']],          # Áo khoác (Clothes)
        'OUMVLP': [['00']]
    }

    gallery_seq_dict = {
        'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],  # Chuỗi gallery cố định
        'OUMVLP': [['01']]
    }


    num_rank = 5  # Đánh giá top-1 đến top-5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])  
    # acc[p, v1, v2, r] lưu rank-r accuracy cho probe_seq p, probe_view v1, gallery_view v2

    # Duyệt qua từng loại chuỗi probe (nm/bg/cl)
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):     # Duyệt qua từng góc quay của probe
                for (v2, gallery_view) in enumerate(view_list):  # Duyệt từng góc quay của gallery

                    # Tạo mask lọc mẫu gallery theo chuỗi và góc quay
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]      # Đặc trưng gallery
                    gallery_y = label[gseq_mask]           # Nhãn gallery

                    # Tạo mask lọc mẫu probe theo chuỗi và góc quay
                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]        # Đặc trưng probe
                    probe_y = label[pseq_mask]             # Nhãn probe


                    # Tính ma trận khoảng cách giữa probe và gallery
                    dist = cuda_dist(probe_x, gallery_x)   # [n_probe, n_gallery]

                    # Sắp xếp khoảng cách tăng dần và lấy chỉ số
                    idx = dist.sort(1)[1].cpu().numpy()    # [n_probe, n_gallery]: index các gallery gần nhất

                    # So sánh nhãn đúng và tính rank-k accuracy
                    acc[p, v1, v2, :] = np.round(
                        np.sum(
                            np.cumsum(                      # Đếm số lần label đúng xuất hiện trong top-k
                                np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1
                            ) > 0, 0) * 100 / dist.shape[0], 2
                    )

    return acc  # Trả về ma trận độ chính xác: acc[probe_type, probe_view, gallery_view, rank]

