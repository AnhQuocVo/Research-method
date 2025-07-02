import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from numpy import ma

df = pd.read_csv(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\after_calculated.csv"
)

# Loại bỏ các dòng thiếu dữ liệu ở 3 biến cần phân tích
df_reg = df[["ln_gdp", "ic_index", "ict_index", "hdi"]].dropna()

# Phân tích PCA để giảm chiều và xem phân tán giữa các quốc gia/năm
features = ["ic_index", "ict_index", "ln_gdp"]
X_std = StandardScaler().fit_transform(df_reg[features])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_std)
df_reg["PC1"] = principalComponents[:, 0]
df_reg["PC2"] = principalComponents[:, 1]

plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    data=df_reg, x="PC1", y="PC2", hue="hdi", palette="coolwarm", alpha=0.8
)
plt.title("PCA of IC, ICT, ln(GDP) colored by HDI")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
norm = plt.Normalize(df_reg["hdi"].min(), df_reg["hdi"].max())
sm_ = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm_.set_array([])
cbar = plt.colorbar(sm_, ax=scatter.axes, label="HDI")
plt.tight_layout()
plt.show()

# Hồi quy tuyến tính đa biến
X = df_reg[["ic_index", "ict_index"]]
X = sm.add_constant(X)
y = df_reg["ln_gdp"]
model = sm.OLS(y, X).fit()
print(model.summary())
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
p = ax.scatter(
    df_reg["ict_index"],
    df_reg["ic_index"],
    df_reg["ln_gdp"],
    c=df_reg["hdi"],
    cmap="coolwarm",
    s=60,
    alpha=0.85,
    edgecolor="k",
)

# Vẽ bề mặt nội suy theo dữ liệu thực
xi = np.linspace(df_reg["ict_index"].min(), df_reg["ict_index"].max(), 30)
yi = np.linspace(df_reg["ic_index"].min(), df_reg["ic_index"].max(), 30)
xi, yi = np.meshgrid(xi, yi)
zi = griddata(
    (df_reg["ict_index"], df_reg["ic_index"]),
    df_reg["ln_gdp"],
    (xi, yi),
    method="cubic",
)
ax.plot_surface(
    xi, yi, zi, color="gray", alpha=0.3, rstride=1, cstride=1, edgecolor="none"
)

# Vẽ phân phối (histogram) lên các mặt phẳng
# 1. Histogram của ict_index trên mặt phẳng (x, z)
hist, bins = np.histogram(df_reg["ict_index"], bins=20)
xcenter = (bins[:-1] + bins[1:]) / 2
ax.bar(
    xcenter,
    hist / hist.max() * (df_reg["ln_gdp"].max() - df_reg["ln_gdp"].min()) * 0.2
    + df_reg["ln_gdp"].min(),
    zs=df_reg["ic_index"].min(),
    zdir="y",
    alpha=0.3,
    color="blue",
    width=(bins[1] - bins[0]),
)

# 2. Histogram của ic_index trên mặt phẳng (y, z)
hist, bins = np.histogram(df_reg["ic_index"], bins=20)
ycenter = (bins[:-1] + bins[1:]) / 2
ax.bar(
    ycenter,
    hist / hist.max() * (df_reg["ln_gdp"].max() - df_reg["ln_gdp"].min()) * 0.2
    + df_reg["ln_gdp"].min(),
    zs=df_reg["ict_index"].min(),
    zdir="x",
    alpha=0.3,
    color="green",
    width=(bins[1] - bins[0]),
)

# 3. Histogram của ln_gdp trên mặt phẳng (x, y)
hist, bins = np.histogram(df_reg["ln_gdp"], bins=20)
zcenter = (bins[:-1] + bins[1:]) / 2
ax.bar3d(
    np.full_like(zcenter, df_reg["ict_index"].max()),
    np.linspace(df_reg["ic_index"].min(), df_reg["ic_index"].max(), len(zcenter)),
    zcenter,
    (df_reg["ict_index"].max() - df_reg["ict_index"].min()) * 0.05,
    (df_reg["ic_index"].max() - df_reg["ic_index"].min()) / len(zcenter),
    hist / hist.max() * (df_reg["ict_index"].max() - df_reg["ict_index"].min()) * 0.2,
    color="red",
    alpha=0.3,
)

cb = fig.colorbar(p, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cb.set_label("HDI")

ax.set_xlabel("ICT Index")
ax.set_ylabel("IC Index")
ax.set_zlabel("ln(GDP)")
ax.set_title(
    "3D Scatter: IC, ICT, ln(GDP) colored by HDI\nwith Interpolated Surface and Distributions"
)
plt.tight_layout()
plt.show()

# Phân tích tương quan
corr = df_reg[["ic_index", "ict_index", "ln_gdp", "hdi"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# Loại bỏ fixed effect theo country
country_avg = df.groupby("country_name")[["ln_gdp", "ic_index", "ict_index"]].transform(
    "mean"
)
df["ln_gdp_demeaned"] = df["ln_gdp"]
df["ic_index_demeaned"] = df["ic_index"]
df["ict_index_demeaned"] = df["ict_index"]
# Loại bỏ các dòng thiếu dữ liệu
df_plot = df[["ict_index_demeaned", "ic_index_demeaned", "ln_gdp_demeaned"]].dropna()

# Định nghĩa x, y, z từ df_plot
x = df_plot["ict_index_demeaned"].values
y = df_plot["ic_index_demeaned"].values
z = df_plot["ln_gdp_demeaned"].values

# Chuẩn bị lưới cho surface (tăng số điểm lưới)
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method="cubic")

# Loại bỏ các vùng NaN ở biên để surface không bị rỗ
zi_masked = ma.masked_invalid(zi)
zi_masked = ma.masked_invalid(zi)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# Vẽ surface mượt hơn
surf = ax.plot_surface(
    xi,
    yi,
    zi_masked,
    cmap="coolwarm",
    alpha=0.9,
    antialiased=True,
    linewidth=0,
    shade=True,
)

# Gán nhãn trục và tiêu đề
ax.set_xlabel("ICT Index")
ax.set_ylabel("IC Index")
ax.set_zlabel("ln(GDP)")
# Thêm colorbar
m = plt.cm.ScalarMappable(cmap="coolwarm")
m.set_array(z)
fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="ln(GDP)")

# Thêm colorbar
m = plt.cm.ScalarMappable(cmap="coolwarm")
m.set_array(z)
fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="hdi")

plt.tight_layout()
plt.show()
