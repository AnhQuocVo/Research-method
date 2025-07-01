import pandas as pd


def clean_country_data(df, time_start=2010, time_end=2023, max_missing=0.2):
    """
    Xử lý dữ liệu quốc gia theo các quy tắc:
    1. Lọc dữ liệu theo khoảng thời gian
    2. Loại bỏ quốc gia có quá nhiều giá trị thiếu cho từng biến

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu với các cột như đã mô tả
    time_start : int, default=2010
        Năm bắt đầu để lọc dữ liệu
    time_end : int, default=2023
        Năm kết thúc để lọc dữ liệu
    max_missing : int, default=2
        Số giá trị thiếu tối đa cho phép cho mỗi biến của mỗi quốc gia

    Returns:
    --------
    pandas.DataFrame
        DataFrame đã được làm sạch
    """

    # Bước 1: Lọc dữ liệu theo khoảng thời gian
    print(f"Bước 1: Lọc dữ liệu từ năm {time_start} đến {time_end}")
    print(f"Số dòng ban đầu: {len(df)}")

    df_filtered = df[(df["year"] >= time_start) & (df["year"] <= time_end)].copy()
    print(f"Số dòng sau khi lọc theo thời gian: {len(df_filtered)}")

    print("Drop columns(OPTION)")
    df_filtered = df_filtered.drop(columns=["p_a"])

    # Kiểm tra số quốc gia duy nhất
    countries = df_filtered["country_name"].unique()
    print(f"Số quốc gia: {len(countries)}")

    # Bước 2: Xác định các biến cần kiểm tra (loại bỏ year, country_name, Country Code)
    variables_to_check = [
        col
        for col in df_filtered.columns
        if col not in ["year", "country_name", "country_code"]
    ]

    print(f"\nBước 2: Kiểm tra missing values cho {len(variables_to_check)} biến")
    print(f"Các biến: {variables_to_check}")

    # Bước 3: Với mỗi biến, loại bỏ quốc gia có quá nhiều giá trị thiếu
    df_cleaned = df_filtered.copy()
    removed_countries_summary = {}

    for variable in variables_to_check:
        print(f"\nĐang xử lý biến: {variable}")

        # Đếm tỷ lệ missing values cho mỗi quốc gia với biến hiện tại
        total_counts = df_cleaned.groupby("country_name")[variable].size()
        missing_counts = df_cleaned.groupby("country_name")[variable].apply(
            lambda x: x.isnull().sum()
        )
        missing_percent = missing_counts / total_counts

        # Tìm các quốc gia có tỷ lệ missing values vượt quá 20%
        countries_to_remove = missing_percent[missing_percent > 0.2].index.tolist()

        if countries_to_remove:
            print(
                f"  Loại bỏ {len(countries_to_remove)} quốc gia có > 20% missing values:"
            )
            for country in countries_to_remove:
                percent = missing_percent[country] * 100
                print(f"    - {country}: {percent:.1f}% missing values")

            # Lưu thông tin quốc gia bị loại bỏ
            removed_countries_summary[variable] = {
                "countries": countries_to_remove,
                "percentages": {
                    country: missing_percent[country] for country in countries_to_remove
                },
            }

            # Loại bỏ các quốc gia này khỏi dataset
            df_cleaned = df_cleaned[
                ~df_cleaned["country_name"].isin(countries_to_remove)
            ]
            print(f"  Số dòng còn lại: {len(df_cleaned)}")
        else:
            print(f"  Không có quốc gia nào bị loại bỏ cho biến {variable}")

    # Bước 4: Tổng kết kết quả
    final_countries = df_cleaned["country_name"].unique()
    print("\n=== TỔNG KẾT ===")
    print(f"Số quốc gia ban đầu: {len(countries)}")
    print(f"Số quốc gia còn lại: {len(final_countries)}")
    print(f"Số dòng dữ liệu cuối cùng: {len(df_cleaned)}")

    # Hiển thị thống kê missing values cuối cùng
    print("\nThống kê missing values sau khi làm sạch:")
    for variable in variables_to_check:
        total_missing = df_cleaned[variable].isnull().sum()
        total_possible = len(df_cleaned)
        missing_percentage = (total_missing / total_possible) * 100
        print(
            f"  {variable}: {total_missing}/{total_possible} ({missing_percentage:.1f}%)"
        )

    return df_cleaned, removed_countries_summary


def analyze_removed_countries(removed_summary, df, numeric_cols):
    print("\n=== PHÂN TÍCH CHI TIẾT CÁC QUỐC GIA BỊ LOẠI BỎ ===")

    if not removed_summary:
        print("Không có quốc gia nào bị loại bỏ!")
        return

    # Đếm số lần mỗi quốc gia bị loại bỏ
    all_removed_countries = []
    for variable, info in removed_summary.items():
        all_removed_countries.extend(info["countries"])

    from collections import Counter

    country_removal_counts = Counter(all_removed_countries)

    print(f"Tổng số quốc gia bị loại bỏ: {len(set(all_removed_countries))}")
    print("\nSố lần mỗi quốc gia bị loại bỏ:")
    for country, count in sorted(
        country_removal_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {country}: {count} lần")

    print("\nChi tiết theo từng biến:")
    for variable, info in removed_summary.items():
        print(f"\n{variable}:")
        for country in info["countries"]:
            missing_percent = info["percentages"][country] * 100
            print(f"  - {country}: {missing_percent:.1f}% missing values")
    print(f"Số dòng còn lại sau khi loại bỏ: {len(df)}")

    """
    Nội suy các giá trị thiếu cho các cột số học trong DataFrame theo từng quốc gia và năm.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame chứa dữ liệu cần nội suy.
    numeric_cols : list
        Danh sách các tên cột số học cần nội suy.

    Returns:
    --------
    pandas.DataFrame
        DataFrame đã được nội suy và loại bỏ các dòng còn thiếu dữ liệu.
    """

    # Đếm số lượng giá trị null trước khi nội suy
    null_before = df[numeric_cols].isnull().sum().sum()

    # Áp dụng nội suy tuyến tính cho từng quốc gia (sắp xếp trong từng group)
    df_interpolated = (
        df.groupby("country_name")
        .apply(
            lambda group: group.sort_values("year").interpolate(
                method="linear", limit_direction="both", axis=0
            )
        )
        .reset_index(drop=True)
    )

    # Đếm số lượng giá trị null sau khi nội suy
    null_after = df_interpolated[numeric_cols].isnull().sum().sum()
    filled_count = null_before - null_after
    print(f"Đã fill nội suy {filled_count} giá trị.")

    # Loại bỏ các dòng còn thiếu dữ liệu sau nội suy
    df_final = df_interpolated.dropna(subset=numeric_cols)
    print(f"Số dòng sau khi loại bỏ các dòng còn thiếu: {len(df_final)}")

    return df_final


def interpolate_missing_values(df, numeric_cols):
    """
    Nội suy các giá trị thiếu cho các cột số học trong DataFrame theo từng quốc gia và năm.
    In ra thông tin về số lượng giá trị thiếu trước/sau khi fill và các cột còn thiếu (nếu có).
    """
    # Đếm số lượng giá trị null trước khi nội suy
    null_before = df[numeric_cols].isnull().sum().sum()
    print(f"Số giá trị thiếu trước khi nội suy: {null_before}")

    # Áp dụng nội suy tuyến tính cho từng quốc gia (sắp xếp trong từng group)
    df_interpolated = (
        df.groupby("country_name")
        .apply(
            lambda group: group.sort_values("year").interpolate(
                method="linear", limit_direction="both", axis=0
            )
        )
        .reset_index(drop=True)
    )

    # Đếm số lượng giá trị null sau khi nội suy
    null_after = df_interpolated[numeric_cols].isnull().sum().sum()
    filled_count = null_before - null_after
    print(f"Đã fill nội suy {filled_count} giá trị.")
    print(f"Số giá trị thiếu còn lại sau khi nội suy: {null_after}")

    # Nếu còn thiếu, in ra các cột còn thiếu và số lượng
    if null_after > 0:
        print("Các cột vẫn còn thiếu dữ liệu sau khi nội suy:")
        print(
            df_interpolated[numeric_cols]
            .isnull()
            .sum()[df_interpolated[numeric_cols].isnull().sum() > 0]
        )

    # Loại bỏ các dòng còn thiếu dữ liệu sau nội suy
    df_final = df_interpolated.dropna(subset=numeric_cols)
    print(f"Số dòng sau khi loại bỏ các dòng còn thiếu: {len(df_final)}")
    print(f"Các cột còn lại trong DataFrame: {list(df_final.columns)}")
    print(f"Số dòng còn lại: {df_final.shape[0]}")

    return df_final


def main(file_path):
    """
    Quy trình chính: đọc dữ liệu, làm sạch và phân tích.
    """
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel")
    print(f"Đã đọc {df.shape[0]} dòng, {df.shape[1]} cột.")

    # Xác định các cột số học (numeric columns) để truyền vào các hàm cần thiết
    numeric_cols = [
        col
        for col in df.columns
        if col not in ["year", "country_name", "Country Code"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Làm sạch và phân tích
    df_cleaned, removed_summary = clean_country_data(df)
    analyze_removed_countries(removed_summary, df, numeric_cols)

    print("Shape cuối cùng sau khi làm sạch:", df_cleaned.shape)
    print(
        "số giá trị thiếu trong DataFrame sau khi làm sạch:",
        df_cleaned.isnull().sum().sum(),
    )

    # Thống kê số missing theo từng cột
    print("\nSố giá trị thiếu theo từng cột sau khi làm sạch:")
    print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])

    return df_cleaned, removed_summary


df_cleaned, removed_summary = main(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\rawdata_2.csv"
)
# Ví dụ sử dụng:
# df_cleaned, removed_summary = main('your_data_file.csv')

# Hoặc nếu bạn đã có DataFrame:
# df_cleaned, removed_summary = clean_country_data(df)
