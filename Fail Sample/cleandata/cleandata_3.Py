import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class CountryFilteringOptimizer:
    def __init__(self, df):
        """
        Khởi tạo với DataFrame chứa dữ liệu
        """
        self.original_df = df.copy()
        self.results = []

    def count_missing_by_country_column(self, df):
        """
        1. Đếm số lượng missing theo từng cột trong từng quốc gia
        """
        countries = df["country_name"].unique()
        data_columns = [
            col
            for col in df.columns
            if col not in ["year", "country_name", "country_code"]
        ]

        missing_summary = []

        for country in countries:
            country_data = df[df["country_name"] == country]
            country_years = len(country_data)

            country_missing = {"country": country, "total_rows": country_years}

            for col in data_columns:
                missing_count = country_data[col].isnull().sum()
                missing_rate = missing_count / country_years * 100
                country_missing[f"{col}_missing_count"] = missing_count
                country_missing[f"{col}_missing_rate"] = round(missing_rate, 2)

            missing_summary.append(country_missing)

        return pd.DataFrame(missing_summary)

    def filter_countries_by_rules(self, df, start_year, end_year=2023):
        """
        2-3. Lọc quốc gia theo các quy tắc:
        - Các cột (trừ p_a): missing <= 20%
        - Cột p_a: missing <= 50% VÀ phải có dữ liệu năm 2023
        """
        # Lọc dữ liệu theo khoảng thời gian
        time_mask = (df["year"] >= start_year) & (df["year"] <= end_year)
        time_df = df[time_mask].copy()

        if len(time_df) == 0:
            return time_df, [], {}

        countries = time_df["country_name"].unique()
        valid_countries = []
        country_details = {}

        # Các cột dữ liệu (trừ year, country_name, country_code)
        data_columns = [
            col
            for col in time_df.columns
            if col not in ["year", "country_name", "country_code"]
        ]
        regular_columns = [col for col in data_columns if col != "p_a"]

        for country in countries:
            country_data = time_df[time_df["country_name"] == country]
            total_rows = len(country_data)
            is_valid = True
            rejection_reasons = []

            # Kiểm tra các cột thường (trừ p_a): missing <= 20%
            for col in regular_columns:
                if col in country_data.columns:
                    missing_count = country_data[col].isnull().sum()
                    missing_rate = missing_count / total_rows

                    if missing_rate > 0.20:  # > 20%
                        is_valid = False
                        rejection_reasons.append(
                            f"{col}: {missing_rate * 100:.1f}% missing"
                        )

            # Kiểm tra cột p_a đặc biệt
            if "p_a" in country_data.columns:
                p_a_missing_count = country_data["p_a"].isnull().sum()
                p_a_missing_rate = p_a_missing_count / total_rows

                # Kiểm tra: missing <= 50%
                if p_a_missing_rate > 0.50:  # > 50%
                    is_valid = False
                    rejection_reasons.append(
                        f"p_a: {p_a_missing_rate * 100:.1f}% missing (>50%)"
                    )

            # Lưu thông tin chi tiết
            country_details[country] = {
                "valid": is_valid,
                "total_rows": total_rows,
                "rejection_reasons": rejection_reasons,
            }

            if is_valid:
                valid_countries.append(country)

        # Lọc DataFrame chỉ giữ các quốc gia hợp lệ
        filtered_df = time_df[time_df["country_name"].isin(valid_countries)].copy()

        return filtered_df, valid_countries, country_details

    def find_optimal_time_range(self):
        """
        4. Lọc dữ liệu từ năm 2010-2023, không duyệt nhiều khoảng năm nữa
        """
        print("🔍 Đang lọc dữ liệu từ năm 2010-2023...")

        start_year = 2010
        end_year = 2023

        filtered_df, valid_countries, country_details = self.filter_countries_by_rules(
            self.original_df, start_year, end_year
        )

        num_countries = len(valid_countries)
        num_rows = len(filtered_df)

        if num_rows > 0:
            data_columns = [
                col
                for col in filtered_df.columns
                if col not in ["year", "country_name", "country_code"]
            ]
            total_cells = len(filtered_df) * len(data_columns)
            missing_cells = filtered_df[data_columns].isnull().sum().sum()
            overall_missing_rate = missing_cells / total_cells * 100
        else:
            overall_missing_rate = 100

        result = {
            "start_year": start_year,
            "end_year": end_year,
            "time_range": f"{start_year}-{end_year}",
            "num_countries": num_countries,
            "num_rows": num_rows,
            "valid_countries": valid_countries,
            "overall_missing_rate": round(overall_missing_rate, 2),
            "filtered_df": filtered_df,
            "country_details": country_details,
        }

        self.results = [result]
        return self.results
        """
        4. Duyệt các khoảng năm từ start_year đến 2023, tìm khoảng tối ưu
        """
        print("🔍 Đang duyệt các khoảng thời gian để tìm khoảng tối ưu...")

        # Tất cả các năm có thể làm start_year (2000-2022)
        available_years = sorted(self.original_df["year"].unique())
        possible_start_years = [year for year in available_years if year <= 2022]

        results = []

        for start_year in possible_start_years:
            # Lọc quốc gia theo quy tắc
            filtered_df, valid_countries, country_details = (
                self.filter_countries_by_rules(self.original_df, start_year, 2023)
            )

            # Tính các chỉ số
            num_countries = len(valid_countries)
            num_rows = len(filtered_df)

            # Tính tỷ lệ missing tổng thể của dữ liệu còn lại
            if num_rows > 0:
                data_columns = [
                    col
                    for col in filtered_df.columns
                    if col not in ["year", "country_name", "country_code"]
                ]
                total_cells = len(filtered_df) * len(data_columns)
                missing_cells = filtered_df[data_columns].isnull().sum().sum()
                overall_missing_rate = missing_cells / total_cells * 100
            else:
                overall_missing_rate = 100

            result = {
                "start_year": start_year,
                "end_year": 2023,
                "time_range": f"{start_year}-2023",
                "num_countries": num_countries,
                "num_rows": num_rows,
                "valid_countries": valid_countries,
                "overall_missing_rate": round(overall_missing_rate, 2),
                "filtered_df": filtered_df,
                "country_details": country_details,
            }

            results.append(result)

            print(
                f"   {start_year}-2023: {num_countries} quốc gia, {num_rows} dòng, missing: {overall_missing_rate:.1f}%"
            )

        # Sắp xếp theo số quốc gia (ưu tiên), sau đó theo tỷ lệ missing thấp
        results.sort(key=lambda x: (-x["num_countries"], x["overall_missing_rate"]))

        self.results = results
        return results

    def get_optimal_result(self):
        """
        Trả về kết quả tối ưu
        """
        if not self.results:
            return None

        optimal = self.results[0]

        # Lấy danh sách các cột còn lại sau lọc
        if len(optimal["filtered_df"]) > 0:
            remaining_columns = [
                col
                for col in optimal["filtered_df"].columns
                if col not in ["year", "country_name", "country_code"]
            ]
        else:
            remaining_columns = []

        return {
            "optimal_time_range": optimal["time_range"],
            "start_year": optimal["start_year"],
            "end_year": optimal["end_year"],
            "num_countries_retained": optimal["num_countries"],
            "countries_retained": optimal["valid_countries"],
            "remaining_columns": remaining_columns,
            "total_rows": optimal["num_rows"],
            "overall_missing_rate": optimal["overall_missing_rate"],
            "filtered_data": optimal["filtered_df"],
        }

    def create_detailed_summary(self):
        """
        Tạo bảng tổng kết chi tiết
        """
        if not self.results:
            return None

        summary_data = []
        for result in self.results:
            summary_data.append(
                {
                    "Khoảng thời gian": result["time_range"],
                    "Năm bắt đầu": result["start_year"],
                    "Số quốc gia": result["num_countries"],
                    "Tổng số dòng": result["num_rows"],
                    "Tỷ lệ missing tổng thể (%)": result["overall_missing_rate"],
                    "Quốc gia giữ lại": ", ".join(result["valid_countries"])
                    if result["valid_countries"]
                    else "Không có",
                }
            )

        return pd.DataFrame(summary_data)

    def analyze_rejection_reasons(self, time_range_result):
        """
        Phân tích lý do từ chối quốc gia
        """
        country_details = time_range_result["country_details"]

        rejected_countries = []
        for country, details in country_details.items():
            if not details["valid"]:
                rejected_countries.append(
                    {
                        "country": country,
                        "reasons": "; ".join(details["rejection_reasons"]),
                    }
                )

        return pd.DataFrame(rejected_countries)

    def run_full_analysis(self):
        """
        Chạy toàn bộ quy trình phân tích
        """
        print("=" * 70)
        print("🚀 BẮT ĐẦU QUY TRÌNH TỐI ỨU HÓA QUỐC GIA")
        print("=" * 70)

        # Tìm khoảng thời gian tối ưu
        self.find_optimal_time_range()

        # Lấy kết quả tối ưu
        optimal_result = self.get_optimal_result()

        if optimal_result is None:
            print("❌ Không tìm thấy khoảng thời gian nào thỏa mãn điều kiện!")
            return None

        print("\n" + "=" * 70)
        print("✅ KẾT QUẢ TỐI ỨU")
        print("=" * 70)
        print(f"🎯 Khoảng thời gian tối ưu: {optimal_result['optimal_time_range']}")
        print(f"📊 Số quốc gia giữ lại: {optimal_result['num_countries_retained']}")
        print(f"📋 Tổng số dòng dữ liệu: {optimal_result['total_rows']}")
        print(f"📉 Tỷ lệ missing tổng thể: {optimal_result['overall_missing_rate']}%")

        print(
            f"\n🌍 Danh sách quốc gia giữ lại ({len(optimal_result['countries_retained'])}):"
        )
        for i, country in enumerate(optimal_result["countries_retained"], 1):
            print(f"   {i}. {country}")

        print(
            f"\n📈 Các cột dữ liệu còn lại ({len(optimal_result['remaining_columns'])}):"
        )
        for i, col in enumerate(optimal_result["remaining_columns"], 1):
            print(f"   {i}. {col}")

        return optimal_result

    def export_results(self, optimal_result, filename_prefix="country_filtering"):
        """
        Xuất kết quả ra file
        """
        if optimal_result is None:
            print("❌ Không có kết quả để xuất!")
            return

        # Xuất dữ liệu đã lọc
        filtered_data = optimal_result["filtered_data"]
        filtered_data.to_excel(f"{filename_prefix}_filtered_data.xlsx", index=False)
        filtered_data.to_csv(f"{filename_prefix}_filtered_data.csv", index=False)

        # Xuất bảng tổng kết
        summary_df = self.create_detailed_summary()
        if summary_df is not None:
            summary_df.to_excel(f"{filename_prefix}_summary.xlsx", index=False)
            summary_df.to_csv(f"{filename_prefix}_summary.csv", index=False)

        # Xuất phân tích lý do từ chối (cho khoảng tối ưu)
        optimal_time_result = self.results[0]
        rejection_df = self.analyze_rejection_reasons(optimal_time_result)
        if len(rejection_df) > 0:
            rejection_df.to_excel(f"{filename_prefix}_rejections.xlsx", index=False)

        print(f"✅ Đã xuất kết quả ra các file: {filename_prefix}_*.xlsx")


# ==================== CÁCH SỬ DỤNG ====================


def main():
    """
    Hàm chính để demo
    """
    # Tạo dữ liệu mẫu để demo
    print("📊 Tạo dữ liệu mẫu để demo...")
    np.random.seed(42)

    countries = [
        "Vietnam",
        "Thailand",
        "Singapore",
        "Malaysia",
        "Indonesia",
        "Philippines",
        "Cambodia",
        "Laos",
        "Myanmar",
        "Brunei",
    ]
    years = list(range(2000, 2024))

    data = []
    for year in years:
        for country in countries:
            # Tạo dữ liệu với pattern missing khác nhau cho mỗi quốc gia
            missing_prob_regular = (
                0.05 if country in ["Singapore", "Malaysia"] else 0.15
            )
            missing_prob_p_a = (
                0.3 if country in ["Cambodia", "Laos", "Myanmar"] else 0.1
            )

            row = {
                "year": year,
                "country_name": country,
                "country_code": country[:3].upper(),
                "gdp": np.random.normal(1000, 500)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "sec_srv": np.random.normal(50, 20)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "mob_sub": np.random.normal(80, 30)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "ter_enr": np.random.normal(30, 15)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "sci_art": np.random.normal(100, 50)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "pop": np.random.normal(50000000, 20000000)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "infl": np.random.normal(3, 2)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "ict_exp": np.random.normal(5, 2)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "hdi": np.random.normal(0.7, 0.1)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "edu_exp": np.random.normal(4, 1)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "fdi": np.random.normal(2, 1)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "trade": np.random.normal(100, 50)
                if np.random.random() > missing_prob_regular
                else np.nan,
                "inet_usr": np.random.normal(60, 20)
                if np.random.random() > missing_prob_regular
                else np.nan,
                # p_a có quy tắc đặc biệt: missing <= 50% và phải có dữ liệu 2023
                "p_a": np.random.normal(1000, 500)
                if (np.random.random() > missing_prob_p_a or year == 2023)
                else np.nan,
            }
            data.append(row)

    df = pd.DataFrame(data)
    print(f"✅ Đã tạo dữ liệu mẫu: {len(df)} dòng, {len(df.columns)} cột")

    # Chạy phân tích
    optimizer = CountryFilteringOptimizer(df)
    optimal_result = optimizer.run_full_analysis()

    # Hiển thị bảng tổng kết
    if optimal_result:
        print("\n📋 BẢNG TỔNG KẾT TẤT CẢ KHOẢNG THỜI GIAN:")
        summary_df = optimizer.create_detailed_summary()
        print(summary_df.head(10).to_string(index=False))

        # Xuất kết quả
        optimizer.export_results(optimal_result, "optimization_results")

    return optimizer, optimal_result


# ==================== SỬ DỤNG VỚI DỮ LIỆU THỰC ====================
df = pd.read_csv(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\rawdata_2.csv"
)


def analyze_real_data(df):
    """
    Hàm để phân tích dữ liệu thực
    """
    optimizer = CountryFilteringOptimizer(df)
    optimal_result = optimizer.run_full_analysis()

    if optimal_result:
        optimizer.export_results(optimal_result, "real_data_analysis")
        return optimal_result
    else:
        return None


# Chạy demo
if __name__ == "__main__":
    optimizer, result = main()

    # Để sử dụng với dữ liệu thực:
    # df = pd.read_csv('your_data.csv')
    # result = analyze_real_data(df)
