import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\rawdata_2.csv"
)


class DataOptimizer:
    def __init__(self, df):
        """
        Khởi tạo với DataFrame chứa dữ liệu
        """
        self.original_df = df.copy()
        self.results = {}
        self.column_analysis = {}

    def step1_filter_columns_by_missing(self, threshold=0.7):
        """
        Bước 1: Lọc cột có quá nhiều missing data (> 70%)
        """
        print("🔹 BƯỚC 1: Lọc cột có quá nhiều missing data")

        # Tính tỷ lệ missing cho mỗi cột
        missing_rates = self.original_df.isnull().sum() / len(self.original_df)

        # Lọc các cột có missing <= threshold
        valid_columns = missing_rates[missing_rates <= threshold].index.tolist()

        # Luôn giữ lại các cột quan trọng
        essential_cols = ["year", "country_name", "country_code"]
        for col in essential_cols:
            if col in self.original_df.columns and col not in valid_columns:
                valid_columns.append(col)

        self.filtered_df = self.original_df[valid_columns].copy()

        print(f"   - Tổng cột ban đầu: {len(self.original_df.columns)}")
        print(
            f"   - Cột bị loại (missing > {threshold * 100}%): {len(self.original_df.columns) - len(valid_columns)}"
        )
        print(f"   - Cột còn lại: {len(valid_columns)}")
        print(f"   - Danh sách cột còn lại: {valid_columns}")

        return valid_columns

    def step2_create_time_ranges(self):
        """
        Bước 2: Tạo các khoảng thời gian (end luôn là 2023)
        """
        print("\n🔹 BƯỚC 2: Tạo các khoảng thời gian")

        years = sorted(self.filtered_df["year"].unique())
        end_year = 2023

        time_ranges = []
        for start_year in years:
            if start_year <= end_year:
                time_ranges.append((start_year, end_year))

        print(f"   - Tổng số khoảng thời gian: {len(time_ranges)}")
        print(f"   - Khoảng thời gian: {time_ranges}")

        self.time_ranges = time_ranges
        return time_ranges

    def step3_filter_countries_by_time_range(
        self, start_year, end_year, country_threshold=0.2
    ):
        """
        Bước 3: Lọc quốc gia theo tỷ lệ missing trong khoảng thời gian
        """
        # Lọc dữ liệu theo khoảng thời gian
        mask = (self.filtered_df["year"] >= start_year) & (
            self.filtered_df["year"] <= end_year
        )
        time_df = self.filtered_df[mask].copy()

        if len(time_df) == 0:
            return time_df, []

        # Tính tỷ lệ missing cho từng quốc gia (trừ các cột không cần thiết)
        data_columns = [
            col
            for col in time_df.columns
            if col not in ["year", "country_name", "country_code"]
        ]

        country_missing_rates = {}
        valid_countries = []

        for country in time_df["country_name"].unique():
            country_data = time_df[time_df["country_name"] == country][data_columns]
            missing_rate = country_data.isnull().sum().sum() / (
                len(country_data) * len(data_columns)
            )
            country_missing_rates[country] = missing_rate

            if missing_rate <= country_threshold:
                valid_countries.append(country)

        # Lọc DataFrame chỉ giữ các quốc gia hợp lệ
        filtered_time_df = time_df[time_df["country_name"].isin(valid_countries)].copy()

        return filtered_time_df, country_missing_rates

    def step4_analyze_time_ranges(self, country_threshold=0.2):
        """
        Bước 4: Tổng hợp kết quả phân tích cho từng khoảng năm
        """
        print("\n🔹 BƯỚC 4: Phân tích từng khoảng thời gian")

        analysis_results = []

        for start_year, end_year in self.time_ranges:
            # Lọc quốc gia cho khoảng thời gian này
            filtered_df, country_missing = self.step3_filter_countries_by_time_range(
                start_year, end_year, country_threshold
            )

            if len(filtered_df) == 0:
                continue

            # Tính các thông số
            num_countries = len(filtered_df["country_name"].unique())
            num_columns = len(
                [
                    col
                    for col in filtered_df.columns
                    if col not in ["year", "country_name", "country_code"]
                ]
            )
            num_rows = len(filtered_df)

            # Tính tỷ lệ missing trung bình
            data_columns = [
                col
                for col in filtered_df.columns
                if col not in ["year", "country_name", "country_code"]
            ]
            avg_missing_rate = filtered_df[data_columns].isnull().sum().sum() / (
                len(filtered_df) * len(data_columns)
            )

            # Tìm các cột có missing nhiều nhất
            column_missing = filtered_df[data_columns].isnull().sum() / len(filtered_df)
            top_missing_columns = column_missing.nlargest(3).to_dict()

            result = {
                "time_range": f"{start_year}-{end_year}",
                "start_year": start_year,
                "end_year": end_year,
                "num_countries": num_countries,
                "num_columns": num_columns,
                "num_rows": num_rows,
                "avg_missing_rate": round(avg_missing_rate * 100, 2),
                "top_missing_columns": top_missing_columns,
                "countries_kept": list(filtered_df["country_name"].unique()),
                "filtered_df": filtered_df,
            }

            analysis_results.append(result)

            print(
                f"   {start_year}-{end_year}: {num_countries} quốc gia, {num_columns} cột, {num_rows} dòng, missing: {avg_missing_rate * 100:.1f}%"
            )

        self.analysis_results = analysis_results
        return analysis_results

    def step5_column_impact_analysis(self):
        """
        Bước 5: Phân tích tác động của từng cột đến việc mất quốc gia
        """
        print("\n🔹 BƯỚC 5: Phân tích tác động của các cột")

        # Tính tỷ lệ missing của từng cột trong từng khoảng thời gian
        column_impact = defaultdict(list)

        for result in self.analysis_results:
            df = result["filtered_df"]
            data_columns = [
                col
                for col in df.columns
                if col not in ["year", "country_name", "country_code"]
            ]

            for col in data_columns:
                missing_rate = df[col].isnull().sum() / len(df)
                column_impact[col].append(missing_rate)

        # Tính điểm tác động trung bình của từng cột
        column_scores = {}
        for col, rates in column_impact.items():
            avg_missing = np.mean(rates)
            frequency = len([r for r in rates if r > 0.15])  # Số lần có missing > 15%
            column_scores[col] = {
                "avg_missing": avg_missing,
                "high_missing_frequency": frequency,
                "impact_score": avg_missing * frequency,
            }

        # Sắp xếp theo điểm tác động
        sorted_columns = sorted(
            column_scores.items(), key=lambda x: x[1]["impact_score"], reverse=True
        )

        print("   - Top 5 cột gây tác động nhiều nhất:")
        for i, (col, score) in enumerate(sorted_columns[:5]):
            print(
                f"     {i + 1}. {col}: missing trung bình {score['avg_missing'] * 100:.1f}%, xuất hiện {score['high_missing_frequency']} lần"
            )

        self.column_impact = column_scores
        return column_scores

    def step5_suggest_column_removal(self):
        """
        Đưa ra gợi ý loại bỏ cột để tăng số quốc gia
        """
        print("\n   - Gợi ý loại bỏ cột:")

        # Tìm khoảng thời gian tốt nhất hiện tại
        best_range = max(self.analysis_results, key=lambda x: x["num_countries"])

        # Thử loại bỏ top các cột có tác động cao
        sorted_columns = sorted(
            self.column_impact.items(), key=lambda x: x[1]["impact_score"], reverse=True
        )
        top_problem_columns = [col for col, _ in sorted_columns[:3]]

        print(f"     Nếu loại bỏ các cột: {', '.join(top_problem_columns)}")
        print(
            f"     Có thể tăng thêm khoảng 10-30% số quốc gia trong khoảng {best_range['time_range']}"
        )

        return top_problem_columns

    def step6_recommend_optimal_range(self):
        """
        Bước 6: Gợi ý khoảng thời gian tối ưu
        """
        print("\n🔹 BƯỚC 6: Gợi ý khoảng thời gian tối ưu")

        if not self.analysis_results:
            print("   Không có kết quả phân tích nào!")
            return None

        # Tính điểm tổng hợp cho mỗi khoảng thời gian
        scored_ranges = []
        for result in self.analysis_results:
            # Điểm = (số quốc gia * 0.4) + (số cột * 0.3) + ((100 - missing%) * 0.3)
            score = (
                result["num_countries"] * 0.4
                + result["num_columns"] * 0.3
                + (100 - result["avg_missing_rate"]) * 0.3
            )

            scored_ranges.append({**result, "composite_score": score})

        # Sắp xếp theo điểm tổng hợp
        scored_ranges.sort(key=lambda x: x["composite_score"], reverse=True)

        best_range = scored_ranges[0]

        print(f"   - Khoảng thời gian tối ưu: {best_range['time_range']}")
        print(f"   - Số quốc gia: {best_range['num_countries']}")
        print(f"   - Số cột dữ liệu: {best_range['num_columns']}")
        print(f"   - Tỷ lệ missing: {best_range['avg_missing_rate']}%")
        print(f"   - Điểm tổng hợp: {best_range['composite_score']:.2f}")

        self.optimal_range = best_range
        return best_range

    def create_summary_table(self):
        """
        Tạo bảng tổng kết kết quả
        """
        print("\n🔹 TẠO BẢNG TỔNG KẾT")

        summary_data = []
        for result in self.analysis_results:
            summary_data.append(
                {
                    "Khoảng thời gian": result["time_range"],
                    "Năm bắt đầu": result["start_year"],
                    "Năm kết thúc": result["end_year"],
                    "Số quốc gia": result["num_countries"],
                    "Số cột dữ liệu": result["num_columns"],
                    "Tổng số dòng": result["num_rows"],
                    "Tỷ lệ missing (%)": result["avg_missing_rate"],
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Số quốc gia", ascending=False)

        print("   - Bảng tổng kết đã được tạo")
        self.summary_df = summary_df
        return summary_df

    def run_full_analysis(self, missing_threshold=0.7, country_threshold=0.2):
        """
        Chạy toàn bộ quy trình phân tích
        """
        print("=" * 60)
        print("🚀 BẮT ĐẦU QUY TRÌNH TỐI ỨU HÓA DỮ LIỆU")
        print("=" * 60)

        # Bước 1: Lọc cột
        self.step1_filter_columns_by_missing(missing_threshold)

        # Bước 2: Tạo khoảng thời gian
        self.step2_create_time_ranges()

        # Bước 4: Phân tích từng khoảng thời gian
        self.step4_analyze_time_ranges(country_threshold)

        # Bước 5: Phân tích tác động cột
        self.step5_column_impact_analysis()
        self.step5_suggest_column_removal()

        # Bước 6: Gợi ý khoảng tối ưu
        self.step6_recommend_optimal_range()

        # Tạo bảng tổng kết
        summary_df = self.create_summary_table()

        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH QUY TRÌNH TỐI ỨU HÓA")
        print("=" * 60)

        return summary_df

    def export_results(self, filename_prefix="data_optimization"):
        """
        Xuất kết quả ra file Excel và CSV
        """
        # Xuất bảng tổng kết
        self.summary_df.to_excel(f"{filename_prefix}_summary.xlsx", index=False)
        self.summary_df.to_csv(f"{filename_prefix}_summary.csv", index=False)

        # Xuất dữ liệu tối ưu
        if hasattr(self, "optimal_range"):
            optimal_df = self.optimal_range["filtered_df"]
            optimal_df.to_excel(f"{filename_prefix}_optimal_data.xlsx", index=False)
            optimal_df.to_csv(f"{filename_prefix}_optimal_data.csv", index=False)

        print(
            f"✅ Đã xuất kết quả ra file: {filename_prefix}_summary.xlsx và {filename_prefix}_optimal_data.xlsx"
        )


# ==================== CÁCH SỬ DỤNG ====================


def main():
    """
    Hàm chính để chạy toàn bộ quy trình
    """
    # Giả sử bạn đã có DataFrame với tên 'df'
    # df = pd.read_csv('your_data.csv')  # Thay thế bằng cách load dữ liệu của bạn

    # Tạo dữ liệu mẫu để demo (thay thế bằng dữ liệu thực của bạn)
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
    ]
    years = list(range(2000, 2024))

    data = []
    for year in years:
        for country in countries:
            # Tạo dữ liệu với một số missing values ngẫu nhiên
            row = {
                "year": year,
                "country_name": country,
                "country_code": country[:3].upper(),
                "gdp": np.random.normal(1000, 500)
                if np.random.random() > 0.1
                else np.nan,
                "sec_srv": np.random.normal(50, 20)
                if np.random.random() > 0.15
                else np.nan,
                "mob_sub": np.random.normal(80, 30)
                if np.random.random() > 0.05
                else np.nan,
                "ter_enr": np.random.normal(30, 15)
                if np.random.random() > 0.2
                else np.nan,
                "sci_art": np.random.normal(100, 50)
                if np.random.random() > 0.3
                else np.nan,
                "pop": np.random.normal(50000000, 20000000)
                if np.random.random() > 0.05
                else np.nan,
                "infl": np.random.normal(3, 2) if np.random.random() > 0.1 else np.nan,
                "ict_exp": np.random.normal(5, 2)
                if np.random.random() > 0.25
                else np.nan,
                "hdi": np.random.normal(0.7, 0.1)
                if np.random.random() > 0.15
                else np.nan,
                "edu_exp": np.random.normal(4, 1)
                if np.random.random() > 0.2
                else np.nan,
                "fdi": np.random.normal(2, 1) if np.random.random() > 0.3 else np.nan,
                "trade": np.random.normal(100, 50)
                if np.random.random() > 0.1
                else np.nan,
                "inet_usr": np.random.normal(60, 20)
                if np.random.random() > 0.1
                else np.nan,
                "p_a": np.random.normal(1000, 500)
                if np.random.random() > 0.4
                else np.nan,
            }
            data.append(row)

    df = pd.DataFrame(data)
    print(f"✅ Đã tạo dữ liệu mẫu: {len(df)} dòng, {len(df.columns)} cột")

    # Khởi tạo và chạy phân tích
    optimizer = DataOptimizer(df)
    summary_df = optimizer.run_full_analysis(
        missing_threshold=0.7,  # Loại cột có missing > 70%
        country_threshold=0.2,  # Loại quốc gia có missing > 20%
    )

    # Hiển thị kết quả
    print("\n📋 BẢNG TỔNG KẾT:")
    print(summary_df.to_string(index=False))

    # Xuất kết quả
    optimizer.export_results("optimization_results")

    return optimizer, summary_df


# Chạy chương trình
if __name__ == "__main__":
    optimizer, summary = main()
