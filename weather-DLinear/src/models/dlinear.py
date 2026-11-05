"""
DLinear: Decomposition-based Linear Model for Time Series Forecasting.
======================================================================

Paper: "Are Transformers Effective for Time Series Forecasting?"
       Zeng et al., AAAI 2023

Y tuong: Thay vi dung Transformer phuc tap, chi can:
  1. Tach chuoi thoi gian thanh Trend + Seasonal bang Moving Average
  2. Moi thanh phan qua 1 lop Linear rieng
  3. Cong lai = du bao

Architecture:
  Input X (batch, L, C_in)
      |
      |--- AvgPool1d(kernel) --> Trend  --> Linear(L->H) --> Y_trend ----
      |                                                                   + --> Output
      |--- X - Trend ---------> Season --> Linear(L->H) --> Y_seasonal --
      |
      +---> Neu C_in != C_out: them Linear(C_in -> C_out) de map so features

DLinear-S (Shared):   1 cap W_trend + W_seasonal dung chung cho TAT CA features
DLinear-I (Individual): Moi feature co W_trend + W_seasonal RIENG
"""

import torch
import torch.nn as nn


# ================================================================
# MovingAvg: Trich xuat TREND bang trung binh truot
# ================================================================

class MovingAvg(nn.Module):
    """
    Trich xuat xu huong dai han bang phep trung binh truot (Moving Average).

    Co che:
      Tai moi thoi diem t, lay trung binh cua k diem xung quanh:
        Trend[t] = (X[t-p] + ... + X[t] + ... + X[t+p]) / k
      voi p = k // 2

      - k cang LON  -> trend cang MUOT, chi giu xu huong chuan
      - k cang NHO  -> trend gan voi original, seasonal yeu

    Padding = replication (giong paper cure-lab/LTSF-Linear):
      Tai 2 dau chuoi, gia tri thieu duoc lap tu diem bien
      (X[0] lap p lan, X[-1] lap p lan) thay vi pad zero.

    Vi du k=25:
      Trend[0]  = (X[0]×13 + X[1] + ... + X[12]) / 25
      Trend[12] = (X[0] + X[1] + ... + X[24]) / 25
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # padding=0 vi da tu pad bang tay trong forward()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        Input:  x (B, C, L)  - batch, channels, sequence length
        Output: trend (B, C, L) - cung shape, da lam min
        """
        p = (self.kernel_size - 1) // 2
        front = x[:, :, 0:1].repeat(1, 1, p)    # lap X[0] p lan
        end = x[:, :, -1:].repeat(1, 1, p)       # lap X[-1] p lan
        x_padded = torch.cat([front, x, end], dim=2)
        return self.avg(x_padded)


# ================================================================
# SeriesDecomp: Tach chuoi thanh Trend + Seasonal
# ================================================================

class SeriesDecomp(nn.Module):
    """
    Phan ra (Decompose) chuoi thoi gian X thanh 2 thanh phan:
      - Trend:    duong muot, xu huong dai han (Moving Average)
      - Seasonal: phan dao dong nhanh quanh trend (X - Trend)

    Seasonal gom: chu ky ngay dem + nhung thay doi dot ngot (mua, may) + nhieu.

    Vi du nhiet do 48h:
      Temp:     [9.4, 9.1, 8.8, ..., 17.4, 18.1]  (len xuong theo ngay dem)
      Trend:    [5.4, 6.1, 6.8, ..., 11.8, 11.4]  (xu huong am dan)
      Seasonal: [+4.0, +3.0, +2.0, ..., +5.6, +6.7] (trua nong +, dem lanh -)
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        """
        Input:  x (B, C, L)
        Output: trend (B, C, L), seasonal (B, C, L)
        """
        trend = self.moving_avg(x)      # lay duong muot
        seasonal = x - trend             # phan con lai = dao dong
        return trend, seasonal


# ================================================================
# DLinear: Mo hinh chinh
# ================================================================

class DLinear(nn.Module):
    """
    DLinear - Decomposition Linear Network.

    LUU Y QUAN TRONG - DLinear-S vs DLinear-I:
      - DLinear-S (individual=False): 1 cap W dung chung cho TAT CA features
        -> 5,899 params, it overfit, nhung cac features khac nhau phai "thoa hiep"
      - DLinear-I (individual=True): moi feature co cap W RIENG
        -> ~70,000 params, hoc tot hon neu features co pattern khac biet
           nhung de overfit hon neu it data

    Args:
        lookback (L):  so gio qua khu (120)
        forecast (H):  so gio du bao (24)
        n_features:    so feature dau vao (12)
        out_features:  so feature dau ra (7) - neu None thi = n_features
        kernel_size:   kich thuoc cua so moving average (25)
        individual:    False = DLinear-S, True = DLinear-I
    """

    def __init__(self, lookback, forecast, n_features, out_features=None,
                 kernel_size=25, individual=False):
        super().__init__()

        self.lookback = lookback
        self.forecast = forecast
        self.n_features = n_features
        self.out_features = out_features or n_features
        self.individual = individual

        # --- Step 0: Decomposition module ---
        # Tach X thanh Trend + Seasonal bang moving average
        self.decomp = SeriesDecomp(kernel_size)

        # --- Step 1: Linear layers cho Trend va Seasonal ---
        # Day la TOAN BO "deep learning" cua model - moi nhom 1 lop Linear don gian

        if individual:
            # DLinear-I: MOI FEATURE co 1 cap Linear RIENG
            # Vi du: temp co W_trend_temp + W_seasonal_temp
            #        cloud co W_trend_cloud + W_seasonal_cloud (KHAC voi cua temp)
            # ModuleList la list cac nn.Module, moi cai doc lap
            self.linear_trend = nn.ModuleList([
                nn.Linear(lookback, forecast) for _ in range(n_features)
                # moi feature: Linear(120 -> 24) = 120*24+24 = 2,904 params
            ])
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(lookback, forecast) for _ in range(n_features)
            ])
        else:
            # DLinear-S: TAT CA FEATURES DUNG CHUNG 1 cap Linear
            # Linear map L (120) diem qua khu -> H (24) diem tuong lai
            # W co shape (24, 120): 24 hang = 24 gio du bao, 120 cot = 120 gio qua khu
            self.linear_trend = nn.Linear(lookback, forecast)
            self.linear_seasonal = nn.Linear(lookback, forecast)
            # moi Linear: 120*24 + 24(bias) = 2,904 params
            # 2 Linear = 5,808 params

        # --- Step 2: Projection layer ---
        # Neu so feature dau vao (12) != dau ra (7), can 1 lop Linear
        # de map tu 12 features -> 7 targets theo protocol
        if self.out_features != n_features:
            self.proj = nn.Linear(n_features, self.out_features)  # 12*7+7 = 91 params
        else:
            self.proj = nn.Identity()  # giu nguyen, khong lam gi

    def forward(self, x):
        """
        Forward pass — toan bo qua trinh du bao.

        Input:
          x: (B, L, C_in) voi B=batch, L=lookback=120, C_in=12 features

        Output:
          y: (B, H, C_out) voi H=forecast=24, C_out=7 targets

        Quy trinh:
          1. Permute (B,L,C) -> (B,C,L) de AvgPool1d hoat dong tren truc thoi gian
          2. Decompose: tach thanh trend + seasonal
          3. Moi thanh phan qua Linear rieng -> du bao 24h
          4. Cong 2 du bao lai
          5. Projection: map tu C_in -> C_out neu can
        """

        # (B, L, C) -> (B, C, L): dua channel len truoc de Conv1d xu ly
        x = x.permute(0, 2, 1)                            # (B, C_in, L)

        # Buoc 1: Decompose
        trend, seasonal = self.decomp(x)                   # ca 2 shape (B, C_in, L)

        # Buoc 2+3: Linear mapping + combine
        if self.individual:
            # DLinear-I: xu ly tung feature doc lap
            t_out = [self.linear_trend[i](trend[:, i, :]) for i in range(self.n_features)]
            s_out = [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(self.n_features)]
            # Moi cai trong list co shape (B, H)
            trend_fc = torch.stack(t_out, dim=2)           # (B, H, C_in)
            seasonal_fc = torch.stack(s_out, dim=2)        # (B, H, C_in)
        else:
            # DLinear-S: tat ca features cung qua 1 Linear
            # Linear(120 -> 24) ap dung tren dim cuoi (time)
            # trend (B, C, L) -> Linear tren dim=-1 -> (B, C, H)
            trend_fc = self.linear_trend(trend)            # (B, C_in, H)
            seasonal_fc = self.linear_seasonal(seasonal)   # (B, C_in, H)
            # Chuyen lai thanh (B, H, C_in)
            trend_fc = trend_fc.permute(0, 2, 1)           # (B, H, C_in)
            seasonal_fc = seasonal_fc.permute(0, 2, 1)     # (B, H, C_in)

        # Buoc 4: Cong 2 nhanh -> du bao cuoi cung
        y = trend_fc + seasonal_fc                         # (B, H, C_in)

        # Buoc 5: Projection C_in -> C_out (12 -> 7 features)
        y = self.proj(y)                                   # (B, H, C_out)

        return y

    def count_params(self):
        """Dem tong so tham so co the hoc duoc."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
