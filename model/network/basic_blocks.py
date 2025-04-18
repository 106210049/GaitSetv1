import torch                          # Thư viện tensor và deep learning
import torch.nn as nn                 # Cung cấp các lớp mạng thần kinh (Conv, Linear,...)
import torch.nn.functional as F       # Các hàm kích hoạt (ReLU, leaky_relu, softmax,...)


class BasicConv2d(nn.Module):                                     # Định nghĩa lớp Convolution 2D đơn giản kế thừa nn.Module
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):  # Nhận số kênh vào, ra, kích thước kernel và các tham số khác như stride, padding
        super(BasicConv2d, self).__init__()                       # Gọi constructor của lớp cha (nn.Module)

        # Tạo lớp convolution 2D không có bias (bias=False), truyền các tham số bổ sung (stride, padding,...)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)


    def forward(self, x):                 # Định nghĩa luồng tính toán
        x = self.conv(x)                  # Truyền input qua lớp Conv2D
        return F.leaky_relu(x, inplace=True)  # Áp dụng LeakyReLU (kích hoạt) trực tiếp lên x



class SetBlock(nn.Module):                              # Định nghĩa khối xử lý dạng set-level (3D CNN)
    def __init__(self, forward_block, pooling=False):   # Nhận 1 block để chạy forward (thường là BasicConv2d), và tuỳ chọn pooling
        super(SetBlock, self).__init__()                # Gọi constructor của lớp cha

        self.forward_block = forward_block              # Lưu lại block sẽ dùng (ví dụ: 1 lớp Conv2D)
        self.pooling = pooling                          # Cờ cho biết có dùng pooling hay không

        if pooling:                                     # Nếu có dùng pooling thì khởi tạo MaxPool2d(kernel_size=2)
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):                   # Định nghĩa luồng tính toán
        n, s, c, h, w = x.size()            # x có shape [batch_size, seq_len, channels, height, width]

        # Chuyển tensor về dạng 2D để đưa qua forward_block: [n * s, c, h, w]
        x = self.forward_block(x.view(-1, c, h, w))

        if self.pooling:                    # Nếu có pooling thì áp dụng sau conv
            x = self.pool2d(x)

        _, c, h, w = x.size()               # Lấy kích thước mới sau conv/pool

        return x.view(n, s, c, h, w)        # Chuyển lại thành tensor 5D gốc: [n, s, c, h, w]

