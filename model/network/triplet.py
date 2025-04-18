import torch                          # Thư viện tensor cho deep learning
import torch.nn as nn                 # Module các lớp mạng thần kinh
import torch.nn.functional as F       # Hàm activation, loss function, v.v.



class TripletLoss(nn.Module):         # Lớp định nghĩa Triplet Loss, kế thừa từ nn.Module
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()       # Gọi constructor của lớp cha
        self.batch_size = batch_size              # Kích thước batch = P * M
        self.margin = margin                      # Biên khoảng cách của triplet loss
        self.hard_or_full = hard_or_full          # Kiểu chọn triplet: 'hard' hoặc 'full'


    def forward(self, feature, label):
        # feature: Tensor [n, m, d] → n = số người (P), m = số mẫu/người (M), d = số chiều đặc trưng
        # label: Tensor [n, m] → nhãn người tương ứng với feature

        n, m, d = feature.size()  # Lấy số người, số mẫu mỗi người, và chiều đặc trưng

        # Tạo mặt nạ:
        # hp_mask: các cặp (i, j) có cùng nhãn (positive pair)
        # hn_mask: các cặp (i, j) khác nhãn (negative pair)
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)  # [n, m, m] → [n*m*m]
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)  # [n, m, m] → [n*m*m]

        # Tính khoảng cách giữa các mẫu trong batch
        dist = self.batch_dist(feature)       # [n, m, m]: ma trận khoảng cách giữa các mẫu
        mean_dist = dist.mean(1).mean(1)      # [n]: khoảng cách trung bình của mỗi người
        dist = dist.view(-1)                  # Flatten về 1 chiều để áp dụng mask

        
        # Hard mining: chọn hardest positive và hardest negative cho mỗi anchor
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]  # [n, m]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]  # [n, m]

        # Tính loss: margin + d(ap) - d(an)
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)  # [n, m]
        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)                           # [n]: loss trung bình mỗi người


        # Full mining: tính loss cho tất cả các tổ hợp (anchor, positive, negative)
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)    # [n, m, P, 1]
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)    # [n, m, 1, N]

        # Tính loss: margin + d(ap) - d(an) với mọi cặp có thể
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)  # [n, m * P * N]

        # Tính loss trung bình cho từng người
        full_loss_metric_sum = full_loss_metric.sum(1)        # [n]: tổng loss của mỗi người
        full_loss_num = (full_loss_metric != 0).sum(1).float()  # [n]: số lượng triplet hợp lệ (loss > 0)
        full_loss_metric_mean = full_loss_metric_sum / full_loss_num  # [n]: loss trung bình mỗi người
        full_loss_metric_mean[full_loss_num == 0] = 0          # Nếu không có triplet hợp lệ → gán 0


        # Trả về:
        # 1. full_loss_metric_mean: loss full triplet (trung bình theo người)
        # 2. hard_loss_metric_mean: loss hard triplet (trung bình theo người)
        # 3. mean_dist: khoảng cách trung bình giữa các mẫu của từng người
        # 4. full_loss_num: số lượng triplet hợp lệ (có đóng góp vào loss)
        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num


    def batch_dist(self, x):
        # x: Tensor [n, m, d] → đặc trưng của batch

        x2 = torch.sum(x ** 2, 2)  # [n, m]: bình phương norm của từng vector

        # Công thức (a - b)^2 = ||a||^2 + ||b||^2 - 2 * a.b
        dist = x2.unsqueeze(2) + x2.unsqueeze(1) - 2 * torch.matmul(x, x.transpose(1, 2))  # [n, m, m]

        dist = torch.sqrt(F.relu(dist))  # Căn bậc hai và đảm bảo không âm
        return dist                      # Trả về ma trận khoảng cách [n, m, m]
