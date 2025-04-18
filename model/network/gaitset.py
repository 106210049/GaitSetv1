import torch                           # Thư viện tensor và tính toán GPU
import torch.nn as nn                  # Cung cấp các lớp mạng thần kinh
import numpy as np                     # Xử lý mảng và tính toán số học

from .basic_blocks import SetBlock, BasicConv2d  # Các khối mạng được định nghĩa riêng



class SetNet(nn.Module):                            # Định nghĩa lớp mạng chính kế thừa từ nn.Module
    def __init__(self, hidden_dim):                 # Khởi tạo với đầu vào là số chiều đặc trưng cuối
        super(SetNet, self).__init__()              # Gọi hàm khởi tạo của lớp cha

        _set_in_channels = 1                        # Ảnh đầu vào là ảnh đen trắng (1 channel)
        _set_channels = [32, 64, 128]               # Kênh trung gian trong các lớp SetBlock

        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))  # Conv5x5
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)  # Conv3x3 + MaxPool
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))  # Conv3x3
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)  # Conv3x3 + MaxPool
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))  # Conv3x3
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))  # Conv3x3

        _gl_in_channels = 32                        # Kênh đầu vào của nhánh global
        _gl_channels = [64, 128]                    # Kênh trung gian trong global layers

        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)           # Downsample không gian ảnh (HxW)


        self.bin_num = [1, 2, 4, 8, 16]              # Chia đặc trưng thành các dải dọc với số lượng bin khác nhau
        self.fc_bin = nn.ParameterList([             # Danh sách các trọng số FC tương ứng cho từng bin
            nn.Parameter(                            # Tạo trọng số learnable (128 input, hidden_dim output)
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))  # *2 vì có 2 nhánh: local và global
        ])


        for m in self.modules():                     # Duyệt qua toàn bộ module con
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):                      # Nếu là Conv
                nn.init.xavier_uniform_(m.weight.data)                    # Khởi tạo Xavier
            elif isinstance(m, nn.Linear):                                 # Nếu là Linear
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):          # Nếu là BatchNorm
                nn.init.normal_(m.weight.data, 1.0, 0.02)                  # mean = 1, std = 0.02
                nn.init.constant_(m.bias.data, 0.0)


    def frame_max(self, x):
        if self.batch_frame is None:                 # Nếu không chia batch → gộp theo trục frame chung
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)  # Gộp frame cho từng đoạn tương ứng từng sequence
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)     # Lấy giá trị max
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0) # Lấy chỉ số max
            return max_list, arg_max_list


    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list


    def forward(self, silho, batch_frame=None):      # silho: [N, T, H, W], batch_frame: số frame từng người

        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()  # Đưa về list trên CPU
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:      # Tìm phần tử cuối khác 0
                    break
                else:
                    _ -= 1                          # Giảm số lượng nếu là 0 (do pad)
            batch_frame = batch_frame[:_]           # Cắt phần 0 ở cuối
            frame_sum = np.sum(batch_frame)         # Tổng frame thực tế
            if frame_sum < silho.size(1):           # Nếu input nhiều hơn frame cần dùng
                silho = silho[:, :frame_sum, :, :]  # Cắt bớt

            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()  # Chuyển về index bắt đầu của mỗi sequence

        n = silho.size(0)                 # Lấy số lượng sequence trong batch
        x = silho.unsqueeze(2)            # Thêm chiều channel → [N, T, 1, H, W]
        del silho                         # Giải phóng bộ nhớ

        # ======== Set-level layers ========
        x = self.set_layer1(x)
        x = self.set_layer2(x)

        gl = self.gl_layer1(self.frame_max(x)[0])  # Nhánh global dùng đặc trưng từ frame_max
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x)[0])  # Cộng đặc trưng global và set-level
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]            # Lấy đặc trưng sau cùng từ set-level
        gl = gl + x                         # Tổng hợp với nhánh global


        feature = list()                    # Danh sách lưu đặc trưng từng bin
        n, c, h, w = gl.size()              # Lấy kích thước đầu ra của gl

        for num_bin in self.bin_num:       # Duyệt qua từng số lượng bin
            z = x.view(n, c, num_bin, -1)             # Chia đặc trưng set-level thành bin
            z = z.mean(3) + z.max(3)[0]               # Pooling: trung bình + max
            feature.append(z)

            z = gl.view(n, c, num_bin, -1)            # Làm tương tự cho nhánh global
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)

        feature = torch.cat(feature, 2)               # Ghép các bin lại theo chiều feature
        feature = feature.permute(2, 0, 1).contiguous()  # [bin, batch, dim]

        feature = feature.matmul(self.fc_bin[0])      # Áp dụng fully-connected cho mỗi bin
        feature = feature.permute(1, 0, 2).contiguous()  # [batch, bin, dim]

        return feature, None                          # Trả về đặc trưng đầu ra và None (label_prob không dùng)

