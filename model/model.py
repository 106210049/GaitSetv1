import math                           # Thư viện toán học, dùng để làm tròn, tính log,...
import os                             # Thao tác với hệ thống file
import os.path as osp                 # Thao tác với đường dẫn file dễ hơn
import random                         # Sinh số và lựa chọn ngẫu nhiên
import sys                            # Tương tác với hệ thống (in stdout, thoát chương trình,...)
from datetime import datetime         # Dùng để đo thời gian (ví dụ: mất bao lâu để train)

import numpy as np                    # Thư viện xử lý ma trận, mảng số
import torch                          # Thư viện deep learning chính
import torch.nn as nn                 # Cung cấp lớp Neural Network cơ bản
import torch.autograd as autograd     # Tự động tính gradient
import torch.optim as optim           # Bộ tối ưu hóa (SGD, Adam,...)
import torch.utils.data as tordata    # Quản lý dataset và dataloader

from .network import TripletLoss, SetNet   # Import mạng đặc trưng (SetNet) và hàm mất mát triplet
from .utils import TripletSampler          # Sampler tạo batch dữ liệu gồm các triplet (anchor, positive, negative)

class Model:
    def __init__(self,
                 hidden_dim,              # Kích thước đặc trưng đầu ra từ encoder
                 lr,                      # Learning rate
                 hard_or_full_trip,       # Chọn 'hard' hoặc 'full' triplet loss
                 margin,                  # Biên margin cho triplet loss
                 num_workers,             # Số luồng load dữ liệu
                 batch_size,              # (P, M): P người, M mẫu mỗi người
                 restore_iter,            # Iteration bắt đầu lại khi resume training
                 total_iter,              # Tổng số iteration muốn huấn luyện
                 save_name,               # Tên mô hình để lưu checkpoint
                 train_pid_num,           # Số lượng người trong tập train
                 frame_num,               # Số frame được chọn trong mỗi sequence
                 model_name,              # Tên mô hình
                 train_source,            # Dataset train
                 test_source,             # Dataset test
                 img_size=64):            # Kích thước ảnh (đã chuẩn hóa)


        # Lưu các tham số cấu hình huấn luyện
        self.save_name = save_name                      # Tên file để lưu checkpoint
        self.train_pid_num = train_pid_num              # Số lượng ID người dùng để huấn luyện
        self.train_source = train_source                # Dataset dùng để train
        self.test_source = test_source                  # Dataset dùng để test

        self.hidden_dim = hidden_dim                    # Số chiều đầu ra của đặc trưng
        self.lr = lr                                    # Learning rate
        self.hard_or_full_trip = hard_or_full_trip      # Chọn 'hard' hoặc 'full' triplet loss
        self.margin = margin                            # Biên margin cho triplet loss
        self.frame_num = frame_num                      # Số frame lấy trong mỗi sequence
        self.num_workers = num_workers                  # Số worker load dữ liệu song song
        self.batch_size = batch_size                    # Tuple (P, M)
        self.model_name = model_name                    # Tên model để nhận diện
        self.P, self.M = batch_size                     # P: số người, M: mẫu mỗi người trong batch

        self.restore_iter = restore_iter                # Số vòng đã train (nếu resume)
        self.total_iter = total_iter                    # Số vòng huấn luyện cần hoàn thành

        self.img_size = img_size                        # Kích thước chuẩn hóa của ảnh đầu vào


        # Khởi tạo encoder và loss, dùng DataParallel cho multi-GPU
        self.encoder = SetNet(self.hidden_dim).float()  # Mạng học đặc trưng SetNet, đầu ra kích thước hidden_dim
        self.encoder = nn.DataParallel(self.encoder)    # Chạy song song trên nhiều GPU nếu có
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)  # Cũng chạy song song
        self.encoder.cuda()                             # Đưa encoder lên GPU
        self.triplet_loss.cuda()                        # Đưa loss lên GPU


        # Bộ tối ưu Adam
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},      # Tối ưu chỉ encoder
        ], lr=self.lr)                                  # Với learning rate được cấu hình


        # Các biến lưu giá trị thống kê trong quá trình huấn luyện
                # Danh sách để lưu giá trị loss và độ lệch trong training
        self.hard_loss_metric = []                      # Loss khi dùng hard triplet
        self.full_loss_metric = []                      # Loss khi dùng full triplet
        self.full_loss_num = []                         # Số lượng triplet hợp lệ trong batch
        self.dist_list = []                             # Danh sách khoảng cách giữa các embedding
        self.mean_dist = 0.01                           # Giá trị trung bình ban đầu của khoảng cách


        self.sample_type = 'all'  # Kiểu lấy frame từ sequence: 'random' hoặc 'all'


    def collate_fn(self, batch):
        batch_size = len(batch)  # Số lượng phần tử trong batch (tức là P * M)
        feature_num = len(batch[0][0])  # Số chiều đặc trưng (thường là 1 với ảnh)

        # Tách thông tin từ batch ra các danh sách riêng
        seqs = [batch[i][0] for i in range(batch_size)]        # Lấy các sequence đặc trưng (list các DataFrame)
        frame_sets = [batch[i][1] for i in range(batch_size)]  # Lấy các danh sách frame tương ứng với mỗi sequence
        view = [batch[i][2] for i in range(batch_size)]        # Lấy góc quay (view) của mỗi sequence
        seq_type = [batch[i][3] for i in range(batch_size)]    # Lấy loại sequence (nm, bg, cl,...)
        label = [batch[i][4] for i in range(batch_size)]       # Lấy ID người của mỗi sequence
        batch = [seqs, view, seq_type, label, None]            # Tạo batch đầu ra tạm thời, phần tử cuối là batch_frame (sẽ cập nhật sau)

        # Hàm phụ: chọn frame cho 1 sequence tại vị trí index
        def select_frame(index):
            sample = seqs[index]           # Một sequence gồm nhiều DataFrame
            frame_set = frame_sets[index]  # Danh sách các frame hợp lệ của sequence này

            if self.sample_type == 'random':  # Nếu chọn khung ngẫu nhiên
                frame_id_list = random.choices(frame_set, k=self.frame_num)  # Chọn ngẫu nhiên self.frame_num frame
                _ = [feature.loc[frame_id_list].values for feature in sample]  # Lấy giá trị các frame đã chọn từ từng đặc trưng
            else:  # Nếu chọn toàn bộ frame có sẵn
                _ = [feature.values for feature in sample]  # Lấy toàn bộ giá trị từ từng đặc trưng
            return _

        # Áp dụng frame sampling cho toàn bộ batch
        seqs = list(map(select_frame, range(len(seqs))))  # seqs giờ là list các đặc trưng đã được sample theo frame

        if self.sample_type == 'random':
            # Với kiểu random: tổ chức dữ liệu theo chiều [feature_num][batch_size][frame,...]
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            # Nếu sample toàn bộ frame → cần xử lý cho multi-GPU và pad frame cho đồng đều
            gpu_num = min(torch.cuda.device_count(), batch_size)       # Số GPU khả dụng (tối đa = batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)            # Số lượng sequence mỗi GPU sẽ xử lý

            # Tạo danh sách số frame từng sample trên mỗi GPU
            batch_frames = [[
                len(frame_sets[i])                                     # Đếm số frame trong mỗi sample
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]

            # Bổ sung 0 nếu batch cuối chưa đủ sample
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)

            # Tính tổng số frame lớn nhất mà mỗi GPU cần pad đến
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])

            # Gộp dữ liệu theo GPU:
            # seqs[j][gpu]: tất cả các đặc trưng thứ j (ảnh, optical flow,...) ghép lại trên 1 GPU
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]

            # Pad tất cả sequence trên mỗi GPU đến độ dài = max_sum_frame
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                    ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),  # Pad thêm frame trắng ở đầu cuối
                    'constant',
                    constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]

            batch[4] = np.asarray(batch_frames)  # batch_frame chứa số lượng frame ban đầu trước khi pad, dùng để cắt lại trong mạng


            batch[0] = seqs       # Gán lại sequence đặc trưng đã chuẩn hóa vào batch[0]
            return batch          # Trả về batch hoàn chỉnh: [features, view, seq_type, label, batch_frame]


    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)  # Nếu đã huấn luyện trước đó, thì load checkpoint tại vòng restore_iter

        self.encoder.train()  # Đặt mô hình ở chế độ huấn luyện (bật dropout, batchnorm, v.v.)
        self.sample_type = 'random'  # Chọn cách lấy frame: ngẫu nhiên frame trong mỗi sequence

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr  # Thiết lập lại learning rate (trường hợp load lại từ checkpoint)

        # Khởi tạo sampler để sinh batch triplet (P người, M mẫu mỗi người)
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)

        # Tạo DataLoader dùng sampler ở trên và collate_fn để xử lý batch
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        # Tập label gốc đã được sắp xếp (giúp ánh xạ nhãn về chỉ số index)
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()  # Ghi lại thời gian bắt đầu để đo thời gian 1000 vòng

        # Vòng lặp qua từng batch dữ liệu
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1  # Tăng số vòng huấn luyện

            self.optimizer.zero_grad()  # Reset gradient trước mỗi batch

            # Chuyển input thành tensor float GPU
            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()  # batch_frame giữ số frame ban đầu trước padding

            # Forward qua encoder để trích xuất đặc trưng và xác suất nhãn (nếu có)
            feature, label_prob = self.encoder(*seq, batch_frame)

            # Biến label string → index trong label_set
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            # Chuẩn bị đặc trưng và nhãn để tính Triplet Loss
            triplet_feature = feature.permute(1, 0, 2).contiguous()  # Hoán đổi (B, C, D) → (C, B, D)
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)  # Lặp label theo chiều batch

            # Tính triplet loss (full + hard) và khoảng cách trung bình giữa embedding
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
            ) = self.triplet_loss(triplet_feature, triplet_label)

            # Chọn loại loss cần tối ưu theo cấu hình
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            # Lưu thống kê loss vào danh sách để log sau
            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:  # Nếu loss đủ lớn (tránh backprop với loss = 0)
                loss.backward()          # Lan truyền gradient
                self.optimizer.step()    # Cập nhật trọng số

            # Sau mỗi 1000 vòng lặp → in thời gian đã chạy
            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            # Sau mỗi 100 vòng lặp → lưu model và in log loss
            if self.restore_iter % 100 == 0:
                self.save()  # Lưu checkpoint
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()  # In ra ngay không bị buffer

                # Reset log sau mỗi lần in
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            # ============ Visualization t-SNE =============
            # Đoạn dưới đây dùng để trực quan hóa đặc trưng bằng t-SNE nhưng đang bị comment
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #     plt.show()

            # Nếu đã đạt đủ số vòng yêu cầu → dừng training
            if self.restore_iter == self.total_iter:
                break


    def ts2var(self, x):
        return autograd.Variable(x).cuda()  # Chuyển tensor x thành một biến autograd có thể tính gradient và đẩy lên GPU


    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))  # Chuyển numpy array thành tensor, sau đó thành Variable trên GPU


    def transform(self, flag, batch_size=1):
        self.encoder.eval()  # Đặt mô hình ở chế độ evaluation (tắt dropout, batchnorm chạy ở chế độ test)

        # Chọn nguồn dữ liệu: test_source hoặc train_source tùy theo flag
        source = self.test_source if flag == 'test' else self.train_source

        self.sample_type = 'all'  # Lấy toàn bộ frame từ mỗi sequence (không random)

        # Tạo DataLoader với sampler tuần tự (không ngẫu nhiên)
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,  # Batch size mặc định là 1, nhưng có thể thay đổi
            sampler=tordata.sampler.SequentialSampler(source),  # Duyệt dữ liệu theo thứ tự
            collate_fn=self.collate_fn,  # Dùng hàm collate để xử lý batch
            num_workers=self.num_workers  # Số luồng load dữ liệu
        )

        # Danh sách lưu trữ output
        feature_list = list()     # Lưu đặc trưng trích xuất từ encoder
        view_list = list()        # Lưu góc quay (view)
        seq_type_list = list()    # Lưu loại sequence (nm, bg, cl,...)
        label_list = list()       # Lưu ID người

        # Duyệt từng batch (thường batch size = 1)
        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x  # Tách các thành phần của batch

            # Chuyển từng đặc trưng sang tensor GPU
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()

            # Chuyển batch_frame sang GPU nếu tồn tại
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            # Trích xuất đặc trưng từ encoder (label_prob không dùng ở đây)
            feature, _ = self.encoder(*seq, batch_frame)

            n, num_bin, _ = feature.size()  # Kích thước đầu ra: (n mẫu, số bin, chiều đặc trưng mỗi bin)

            # Chuyển feature thành dạng numpy, gộp theo batch
            feature_list.append(feature.view(n, -1).data.cpu().numpy())  # (n, num_bin * dim)

            # Lưu thông tin phụ trợ để đánh giá
            view_list += view
            seq_type_list += seq_type
            label_list += label

        # Nối tất cả đặc trưng lại thành 1 mảng (N x D)
        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list


    def save(self):
        # Tạo thư mục lưu model nếu chưa tồn tại
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)

        # Lưu encoder (trọng số mạng học đặc trưng)
        torch.save(self.encoder.state_dict(),
                osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(  # Tên file: save_name-iteration-encoder.ptm
                                self.save_name, self.restore_iter)))

        # Lưu trạng thái optimizer (Adam), để khi resume training thì tiếp tục như cũ
        torch.save(self.optimizer.state_dict(),
                osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(  # Tên file: save_name-iteration-optimizer.ptm
                                self.save_name, self.restore_iter)))


    # restore_iter: iteration index of the checkpoint to load
    # Hàm load lại mô hình từ checkpoint đã lưu
# restore_iter: số vòng lặp mà checkpoint đã lưu
def load(self, restore_iter):
    # Load trọng số mạng encoder
    self.encoder.load_state_dict(torch.load(osp.join(
        'checkpoint', self.model_name,
        '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))

    # Load trạng thái optimizer (để tiếp tục train mà không mất momentum, v.v.)
    self.optimizer.load_state_dict(torch.load(osp.join(
        'checkpoint', self.model_name,
        '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
