"""RoCA의 CNN + LSTM encoder/decoder backbone.

입력 shape:
    (B, C, T)

흐름:
    Conv1d blocks
        ↓
    시간축 feature sequence
        ↓
    LSTM Encoder
        ↓
    LSTM Decoder로 역방향 복원
        ↓
    원본 hidden / 복원 output flatten
        ↓
    동일한 projection head
        ↓
    project, rec_project

최종 anomaly score 자체는 이 파일에서 계산하지 않는다.
두 projection과 hypersphere center 사이의 거리는 loss.py의 roca_loss()에서 계산한다.
"""

from torch import nn
import torch

import math
class base_Model(nn.Module):
    """RoCA 원본의 기본 backbone.

    Conv1d가 지역적인 시간 패턴을 추출하고,
    LSTM encoder/decoder가 window 내부의 순차 구조를 압축하고 복원한다.
    마지막 projection_head는 원본 표현과 복원 표현을 같은 공간으로 보낸다.
    """

    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.dataset = configs.dataset
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        # 원본 구현은 features_len을 설정값으로 받았지만,
        # paperib에서는 아래 dummy forward로 실제 CNN 출력 길이를 자동 계산한다.
        self.project_channels = configs.project_channels
        self.hidden_size = configs.hidden_size
        self.window_size = configs.window_size
        self.device = device
        self.num_layers = configs.num_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        # 원본 RoCA는 데이터셋별로 features_len을 수동 설정했다.
        # paperib는 window_size가 달라져도 Linear layer 크기가 깨지지 않도록
        # dummy 입력을 CNN에 통과시켜 실제 출력 길이를 다시 계산한다.
        if self.dataset == 'IOpsCompetition' or self.dataset == 'UCR':
            self.features_len = math.floor(0.25 * self.window_size / self.stride) + 2
        else:
            self.features_len = math.floor(math.floor(0.125 * self.window_size / self.stride) + 5 / 2)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # 실제 입력과 같은 (B=1, C, T) dummy tensor를 만들어
        # CNN 통과 후 남는 시간축 길이를 측정한다.
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.window_size)
            x = self.conv_block1(dummy)
            if self.dataset == 'IOpsCompetition' or self.dataset == 'UCR':
                x = self.conv_block2(x)
            else:
                x = self.conv_block3(x)
                x = self.conv_block4(x)
            self.features_len = int(x.shape[-1])

        self.encoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.final_out_channels)
        self.project = nn.Linear(self.final_out_channels * self.features_len, self.project_channels, bias=False)
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, self.project_channels),
        )

    def init_hidden_state(self, batch_size):
        """LSTM용 0 초기 hidden/cell state를 만든다.

        현재 forward에서는 PyTorch LSTM의 기본 0 초기화를 사용하므로
        이 함수는 직접 호출되지 않지만 원본 구조를 이해하기 위해 유지한다.
        """

        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_in):
        """한 window batch를 project와 rec_project 두 representation으로 변환한다."""

        if torch.isnan(x_in).any():
            print('tensor contain nan')

        # 1단계: Conv1d로 local temporal pattern을 추출한다.

        x = self.conv_block1(x_in)
        if self.dataset == 'IOpsCompetition' or self.dataset == 'UCR':
            x = self.conv_block2(x)
        else:
            x = self.conv_block3(x)
            x = self.conv_block4(x)
        # 2단계: Conv1d 출력은 (B, C, T)이므로
        # LSTM 입력 형식 (B, T, C)에 맞게 축을 바꾼다.
        hidden = x.permute(0, 2, 1)

        # Encoder는 전체 window를 읽고 마지막 hidden/cell state를 만든다.
        # 아래 주석 코드는 명시적인 0 state를 넣던 원본 대안이며 현재는 사용하지 않는다.
        # enc_hidden = self.init_hidden_state(hidden.shape[0])
        _, enc_hidden = self.encoder(hidden)
        # _, enc_hidden = self.encoder(hidden, enc_hidden)

        # 3단계: Decoder는 encoder의 마지막 state에서 시작한다.
        dec_hidden = enc_hidden
        output = torch.zeros(hidden.shape).to(self.device)

        # Decoder는 시간축을 뒤에서 앞으로 복원한다.
        # 학습 시에는 실제 hidden을 다음 입력으로 넣는 teacher-forcing 형태이고,
        # 평가 시에는 직전에 생성한 output을 다시 입력으로 사용한다.
        for i in reversed(range(hidden.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(
                    hidden[:, i].unsqueeze(1).float(),
                    dec_hidden,
                )
            else:
                _, dec_hidden = self.decoder(
                    output[:, i].unsqueeze(1),
                    dec_hidden,
                )
        # 4단계: 원본 encoder sequence와 decoder reconstruction sequence를
        # 각각 flatten한 뒤 같은 projection head에 통과시킨다.
        hidden = hidden.reshape(hidden.size(0), -1)
        output = output.reshape(output.size(0), -1)
        project = self.projection_head(hidden)
        rec_project = self.projection_head(output)

        # loss.py는 이 두 vector가 hypersphere center와 얼마나 떨어졌는지 계산한다.
        return project, rec_project

