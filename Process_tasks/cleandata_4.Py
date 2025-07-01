import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics.pairwise import nan_euclidean_distances

warnings.filterwarnings("ignore")


def process_dataset(df):
    """
    Xử lý dataset theo các yêu cầu:
    1. Filter 2010-2023
    2. Đếm missing values theo quốc gia và cột
    3. Loại bỏ quốc gia có >20% missing trên bất kỳ cột nào
    4. Fill missing values theo chiều dọc (nội suy)
    5. Fill missing values theo chiều ngang (KNN - optional)
    """

    print("=== BƯỚC 1: FILTER DỮ LIỆU 2010-2023 ===")
    # Filter data từ 2010-2023
    df_filtered = df[(df["year"] >= 2010) & (df["year"] <= 2023)].copy()

    # Đếm số dòng trước khi loại bỏ duplicate
    before_drop_duplicates = len(df_filtered)
    # Xóa duplicate row (giữ lại dòng đầu tiên)
    df_filtered = df_filtered.drop_duplicates()
    after_drop_duplicates = len(df_filtered)
    print(f"Số dòng sau khi filter: {after_drop_duplicates}")
    print(f"Số quốc gia: {df_filtered['country_name'].nunique()}")
    print(f"Năm từ {df_filtered['year'].min()} đến {df_filtered['year'].max()}")
    print(
        "Số dòng Duplicate đã loại bỏ:", before_drop_duplicates - after_drop_duplicates
    )

    # === BƯỚC 1.1 (OPTION): LỌC BỚT CỘT ===
    print("=== BƯỚC 1.1 (OPTION): LỌC BỚT CỘT ===")
    drop_cols = ["rnd_exp", "p_a", "hte"]
    df_filtered = df_filtered.drop(columns=drop_cols, errors="ignore")
    print(f"Đã loại bỏ các cột: {drop_cols}")

    # Các cột numeric cần kiểm tra missing (chỉ lấy các cột còn lại)
    # Lấy tất cả các cột numeric (trừ country_name, country_code, year)
    exclude_cols = ["country_name", "year"]
    for col in df_filtered.columns:
        if col not in exclude_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")
    all_numeric_cols = [col for col in df_filtered.columns if col not in exclude_cols]
    numeric_cols = [col for col in all_numeric_cols if col in df_filtered.columns]

    # Đếm missing values theo quốc gia cho từng cột
    missing_by_country = {}
    for country in df_filtered["country_name"].unique():
        country_data = df_filtered[df_filtered["country_name"] == country]
        total_years = len(country_data)

        missing_count = {}
        missing_pct = {}
        for col in numeric_cols:
            missing = country_data[col].isna().sum()
            missing_count[col] = missing
            missing_pct[col] = (missing / total_years) * 100

        missing_by_country[country] = {
            "total_years": total_years,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }

    # Tạo DataFrame để hiển thị missing percentages
    missing_df = pd.DataFrame(
        {country: data["missing_pct"] for country, data in missing_by_country.items()}
    ).T

    print("Top 10 quốc gia có nhiều missing data nhất (% missing trung bình):")
    avg_missing = missing_df.mean(axis=1).sort_values(ascending=False)
    print(avg_missing.head(10))

    print("\n=== BƯỚC 3: LOẠI BỎ QUỐC GIA CÓ >20% MISSING ===")
    # Tìm quốc gia có bất kỳ cột nào missing >20%
    countries_to_remove = []
    col_remove_count = {
        col: 0 for col in numeric_cols
    }  # Thống kê số quốc gia bị loại bởi từng cột

    for country, data in missing_by_country.items():
        max_missing_pct = max(data["missing_pct"].values())
        if max_missing_pct > 20:
            countries_to_remove.append((country, max_missing_pct))
            # Đếm cho từng cột
            for col, pct in data["missing_pct"].items():
                if pct > 20:
                    col_remove_count[col] += 1

    # Sắp xếp theo thứ tự từ nhiều đến ít missing
    countries_to_remove.sort(key=lambda x: x[1], reverse=True)

    print(f"Số quốc gia bị loại bỏ: {len(countries_to_remove)}")
    if countries_to_remove:
        print("Danh sách quốc gia bị loại (theo thứ tự % missing giảm dần):")
        for country, pct in countries_to_remove[:10]:  # Hiển thị 10 đầu
            print(f"  {country}: {pct:.1f}% missing")

    # Thống kê số quốc gia bị loại bởi từng cột
    print("\nSố quốc gia bị loại bởi từng cột (có >20% missing):")
    for col in numeric_cols:
        print(f"  {col}: {col_remove_count[col]} quốc gia")

    # Lọc bỏ các quốc gia này
    countries_to_keep = [
        country
        for country, _ in missing_by_country.items()
        if country not in [c[0] for c in countries_to_remove]
    ]

    df_clean = df_filtered[df_filtered["country_name"].isin(countries_to_keep)].copy()
    print(f"Số quốc gia còn lại: {len(countries_to_keep)}")
    print(f"Số dòng còn lại: {len(df_clean)}")
    print(
        f"Tổng missing values sau khi loại bỏ: {df_clean[numeric_cols].isna().sum().sum()}"
    )

    print("\n=== BƯỚC 4: FILL MISSING VALUES THEO CHIỀU DỌC (NỘI SUY) ===")
    # Fill missing values theo từng quốc gia (nội suy theo thời gian)
    df_interpolated = df_clean.copy()

    fill_stats = {}  # Lưu số lượng và tỷ lệ fill cho từng quốc gia

    for country in df_interpolated["country_name"].unique():
        mask = df_interpolated["country_name"] == country
        country_data = df_interpolated[mask].copy()

        # Sắp xếp theo năm
        country_data = country_data.sort_values("year")

        # Đếm missing trước khi nội suy
        missing_before = country_data[numeric_cols].isna().sum().sum()

        # Nội suy cho tất cả các cột numeric cùng lúc
        country_data[numeric_cols] = country_data[numeric_cols].interpolate(
            method="linear", axis=0
        )

        # Đếm missing sau khi nội suy
        missing_after = country_data[numeric_cols].isna().sum().sum()

        # Số lượng giá trị đã fill
        filled_count = missing_before - missing_after
        total_values = len(country_data) * len(numeric_cols)
        fill_ratio = filled_count / total_values * 100 if total_values > 0 else 0

        fill_stats[country] = {
            "filled_count": filled_count,
            "fill_ratio_pct": fill_ratio,
            "total_values": total_values,
        }

        df_interpolated.loc[mask, country_data.columns] = country_data

    # Hiển thị top 10 quốc gia có tỷ lệ fill cao nhất
    fill_stats_df = pd.DataFrame(fill_stats).T
    fill_stats_df = fill_stats_df.sort_values("fill_ratio_pct", ascending=False)
    print("\nTop 10 quốc gia có tỷ lệ giá trị được fill cao nhất sau nội suy:")
    print(fill_stats_df[["filled_count", "fill_ratio_pct"]].head(10))

    # Kiểm tra missing sau nội suy
    missing_after_interp = df_interpolated[numeric_cols].isna().sum().sum()
    print(f"Tổng missing values sau nội suy: {missing_after_interp}")

    print("\n=== BƯỚC 5: FILL MISSING VALUES THEO CHIỀU NGANG (KNN - OPTIONAL) ===")
    df_final = df_interpolated.copy()

    knn_fill_stats = {}  # Lưu số lượng và tỷ lệ fill KNN cho từng quốc gia
    knn_used_countries = {}  # Lưu quốc gia đã dùng để fill cho từng quốc gia

    if missing_after_interp > 0:
        print("Áp dụng KNN Imputer cho missing values còn lại...")

        feature_cols = [
            col
            for col in numeric_cols
            if df_final[col].isna().sum() < len(df_final) * 0.5
        ]

        if len(feature_cols) >= 3:
            for year in df_final["year"].unique():
                year_mask = df_final["year"] == year
                year_data = df_final[year_mask].copy()

                if year_data[numeric_cols].isna().any().any():
                    scaler = StandardScaler()
                    available_data = year_data[feature_cols].dropna()

                    if len(available_data) >= 3:
                        scaler.fit(available_data)
                        scaler.transform(
                            year_data[feature_cols].fillna(
                                year_data[feature_cols].mean()
                            )
                        )

                        knn_imputer = KNNImputer(
                            n_neighbors=min(5, len(available_data))
                        )

                        numeric_data = year_data[numeric_cols].values
                        missing_before = (
                            pd.DataFrame(numeric_data, columns=numeric_cols)
                            .isna()
                            .sum(axis=1)
                        )

                        # Lưu lại index các quốc gia để truy vết
                        country_names = year_data["country_name"].tolist()

                        imputed_data = knn_imputer.fit_transform(numeric_data)

                        # Lấy thông tin các quốc gia được dùng để fill cho từng quốc gia
                        # KNNImputer dùng khoảng cách Euclidean, ta lấy nearest neighbors index
                        distances = nan_euclidean_distances(numeric_data, numeric_data)
                        n_neighbors = min(5, len(available_data))
                        for idx, row in enumerate(year_data.itertuples()):
                            country = getattr(row, "country_name")
                            before = missing_before.iloc[idx]
                            after = pd.isna(imputed_data[idx]).sum()
                            filled = int(before - after)
                            if country not in knn_fill_stats:
                                knn_fill_stats[country] = {
                                    "filled_count": 0,
                                    "total_values": 0,
                                }
                                knn_used_countries[country] = set()
                            knn_fill_stats[country]["filled_count"] += filled
                            knn_fill_stats[country]["total_values"] += len(numeric_cols)
                            # Lấy index các nước gần nhất (loại bỏ chính nó)
                            neighbor_idx = distances[idx].argsort()[1 : n_neighbors + 1]
                            for nidx in neighbor_idx:
                                knn_used_countries[country].add(country_names[nidx])

                        df_final.loc[year_mask, numeric_cols] = imputed_data

        missing_after_knn = df_final[numeric_cols].isna().sum().sum()
        print(f"Tổng missing values sau KNN: {missing_after_knn}")

        # Tính tỷ lệ fill KNN cho từng quốc gia
        for country in knn_fill_stats:
            total = knn_fill_stats[country]["total_values"]
            filled = knn_fill_stats[country]["filled_count"]
            knn_fill_stats[country]["fill_ratio_pct"] = (
                (filled / total * 100) if total > 0 else 0
            )
            # Thêm danh sách quốc gia đã dùng để fill
            knn_fill_stats[country]["used_countries"] = ", ".join(
                sorted(knn_used_countries.get(country, []))
            )

        # Hiển thị top 10 quốc gia có tỷ lệ fill KNN cao nhất
        knn_fill_stats_df = pd.DataFrame(knn_fill_stats).T
        knn_fill_stats_df = knn_fill_stats_df.sort_values(
            "fill_ratio_pct", ascending=False
        )
        print("\nTop 10 quốc gia có tỷ lệ giá trị được fill bằng KNN cao nhất:")
        print(
            knn_fill_stats_df[
                ["filled_count", "fill_ratio_pct", "used_countries"]
            ].head(10)
        )

    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(f"Số quốc gia: {df_final['country_name'].nunique()}")
    print(f"Số năm: {df_final['year'].nunique()}")
    print(f"Tổng số dòng: {len(df_final)}")
    print(f"Tổng missing values: {df_final[numeric_cols].isna().sum().sum()}")

    # Hiển thị missing values theo cột
    print("\nMissing values theo từng cột:")
    for col in numeric_cols:
        missing = df_final[col].isna().sum()
        pct = (missing / len(df_final)) * 100
        print(f"  {col}: {missing} ({pct:.2f}%)")

    print("Missing Valuse còn lại", df_final[numeric_cols].isna().sum().sum())

    # Danh sách các quốc gia còn lại
    print("\nDanh sách country_name còn lại sau xử lý:")
    print(sorted(df_final["country_name"].unique()))

    return df_final, missing_by_country, countries_to_remove


df = pd.read_csv(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\rawdata_2,1.csv"
)
df_processed, missing_stats, removed_countries = process_dataset(df)

print(df_processed.shape)

# Sử dụng hàm
# df = pd.read_csv('your_dataset.csv')  # Uncomment và thay đổi path
# df_processed, missing_stats, removed_countries = process_dataset(df)

# Lưu kết quả
# df_processed.to_csv('processed_dataset.csv', index=False)

print("Code đã sẵn sàng! Uncomment các dòng cuối để chạy với dataset của bạn.")
