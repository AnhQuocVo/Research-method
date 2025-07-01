import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.regression.linear_model import OLS
import warnings

warnings.filterwarnings("ignore")


class PanelStationarityTests:
    """
    Class để thực hiện các kiểm định tính dừng và đồng liên kết cho panel data
    """

    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = [
            col
            for col in df.columns
            if col not in ["Time", "Country Name", "Country Code"]
        ]
        self.results = {}

    def prepare_data(self):
        """Chuẩn bị dữ liệu cho phân tích"""
        print("=== CHUẨN BỊ DỮ LIỆU ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Số quốc gia: {self.df['Country Name'].nunique()}")
        print(f"Khoảng thời gian: {self.df['Time'].min()} - {self.df['Time'].max()}")
        print(f"Biến số: {self.numeric_cols}")
        print("-" * 60)

        # Kiểm tra missing values
        missing_info = self.df[self.numeric_cols].isnull().sum()
        if missing_info.sum() > 0:
            print("Missing values detected:")
            for col, missing in missing_info.items():
                if missing > 0:
                    print(f"  {col}: {missing} ({missing / len(self.df):.2%})")
        else:
            print("Không có missing values")
        print("-" * 60)

    def unit_root_tests_individual(self, max_countries_display=5):
        """
        Thực hiện kiểm định unit root cho từng quốc gia và từng biến
        """
        print("=== KIỂM ĐỊNH UNIT ROOT CHO TỪNG QUỐC gia ===")

        countries = self.df["Country Name"].unique()
        self.results["individual_tests"] = {}

        for var in self.numeric_cols:
            print(f"\n--- Biến: {var} ---")
            self.results["individual_tests"][var] = {}

            adf_results = []
            kpss_results = []

            valid_countries = 0

            for country in countries:
                country_data = self.df[self.df["Country Name"] == country][var].dropna()

                if len(country_data) < 10:  # Cần ít nhất 10 quan sát
                    continue

                valid_countries += 1

                try:
                    # ADF Test
                    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(
                        country_data, autolag="AIC"
                    )
                    adf_results.append(
                        {
                            "country": country,
                            "statistic": adf_stat,
                            "p_value": adf_pval,
                            "stationary": adf_pval < 0.05,
                        }
                    )

                    # KPSS Test
                    kpss_stat, kpss_pval, _, kpss_crit = kpss(
                        country_data, regression="c"
                    )
                    kpss_results.append(
                        {
                            "country": country,
                            "statistic": kpss_stat,
                            "p_value": kpss_pval,
                            "stationary": kpss_pval > 0.05,  # KPSS: H0 là stationary
                        }
                    )

                except Exception:
                    continue

            # Lưu kết quả
            self.results["individual_tests"][var] = {
                "adf": adf_results,
                "kpss": kpss_results,
            }

            # Tóm tắt kết quả
            if adf_results:
                adf_stationary = sum([r["stationary"] for r in adf_results])
                print(
                    f"ADF Test: {adf_stationary}/{len(adf_results)} quốc gia có chuỗi dừng"
                )

                # Hiển thị một số quốc gia mẫu
                print("  Mẫu kết quả ADF:")
                for i, result in enumerate(adf_results[:max_countries_display]):
                    status = "Dừng" if result["stationary"] else "Không dừng"
                    print(
                        f"    {result['country']}: {result['statistic']:.3f} (p={result['p_value']:.3f}) - {status}"
                    )

            if kpss_results:
                kpss_stationary = sum([r["stationary"] for r in kpss_results])
                print(
                    f"KPSS Test: {kpss_stationary}/{len(kpss_results)} quốc gia có chuỗi dừng"
                )

                print("  Mẫu kết quả KPSS:")
                for i, result in enumerate(kpss_results[:max_countries_display]):
                    status = "Dừng" if result["stationary"] else "Không dừng"
                    print(
                        f"    {result['country']}: {result['statistic']:.3f} (p={result['p_value']:.3f}) - {status}"
                    )

        print("-" * 60)

    def panel_unit_root_summary(self):
        """
        Tóm tắt kết quả kiểm định unit root cho toàn panel
        """
        print("=== TÓM TẮT KIỂM ĐỊNH UNIT ROOT PANEL ===")

        summary_data = []

        for var in self.numeric_cols:
            if var in self.results["individual_tests"]:
                adf_results = self.results["individual_tests"][var]["adf"]
                kpss_results = self.results["individual_tests"][var]["kpss"]

                if adf_results and kpss_results:
                    adf_stationary_pct = (
                        sum([r["stationary"] for r in adf_results])
                        / len(adf_results)
                        * 100
                    )
                    kpss_stationary_pct = (
                        sum([r["stationary"] for r in kpss_results])
                        / len(kpss_results)
                        * 100
                    )

                    # Consensus: cả hai test đồng ý
                    consensus_stationary = 0
                    total_consensus = 0

                    for adf_r, kpss_r in zip(adf_results, kpss_results):
                        if adf_r["country"] == kpss_r["country"]:
                            total_consensus += 1
                            if adf_r["stationary"] and kpss_r["stationary"]:
                                consensus_stationary += 1

                    consensus_pct = (
                        (consensus_stationary / total_consensus * 100)
                        if total_consensus > 0
                        else 0
                    )

                    summary_data.append(
                        {
                            "Variable": var,
                            "ADF_Stationary_%": f"{adf_stationary_pct:.1f}%",
                            "KPSS_Stationary_%": f"{kpss_stationary_pct:.1f}%",
                            "Consensus_Stationary_%": f"{consensus_pct:.1f}%",
                            "Recommendation": "Dừng"
                            if consensus_pct > 50
                            else "Không dừng/Cần differencing",
                        }
                    )

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print("-" * 60)

        return summary_df

    def first_difference_tests(self):
        """
        Kiểm định tính dừng cho chuỗi sai phân bậc 1
        """
        print("=== KIỂM ĐỊNH TÍNH DỪNG SAI PHÂN BẬC 1 ===")

        self.results["first_diff_tests"] = {}

        for var in self.numeric_cols:
            print(f"\n--- Sai phân bậc 1 của {var} ---")

            adf_results = []
            countries = self.df["Country Name"].unique()

            for country in countries:
                country_data = self.df[self.df["Country Name"] == country][var].dropna()

                if len(country_data) < 12:  # Cần nhiều hơn cho sai phân
                    continue

                # Tính sai phân bậc 1
                diff_data = country_data.diff().dropna()

                if len(diff_data) < 10:
                    continue

                try:
                    # ADF Test cho sai phân
                    adf_stat, adf_pval, _, _, _, _ = adfuller(diff_data, autolag="AIC")
                    adf_results.append(
                        {
                            "country": country,
                            "statistic": adf_stat,
                            "p_value": adf_pval,
                            "stationary": adf_pval < 0.05,
                        }
                    )
                except Exception:
                    continue

            self.results["first_diff_tests"][var] = adf_results

            if adf_results:
                stationary_count = sum([r["stationary"] for r in adf_results])
                stationary_pct = stationary_count / len(adf_results) * 100
                print(
                    f"Sai phân bậc 1: {stationary_count}/{len(adf_results)} ({stationary_pct:.1f}%) quốc gia có chuỗi dừng"
                )

                # Mẫu kết quả
                print("  Mẫu kết quả:")
                for result in adf_results[:3]:
                    status = "Dừng" if result["stationary"] else "Không dừng"
                    print(
                        f"    {result['country']}: {result['statistic']:.3f} (p={result['p_value']:.3f}) - {status}"
                    )

        print("-" * 60)

    def lag_selection(self, max_lags=4):
        """
        Xác định cấu trúc độ trễ phù hợp
        """
        print("=== XÁC ĐỊNH CẤU TRÚC ĐỘ TRỄ ===")

        self.results["lag_selection"] = {}

        # Tính correlation matrix để xác định mối quan hệ
        corr_matrix = self.df[self.numeric_cols].corr()

        print("Ma trận tương quan giữa các biến:")
        print(corr_matrix.round(3))
        print()

        # Tìm các cặp biến có correlation cao
        high_corr_pairs = []
        for i in range(len(self.numeric_cols)):
            for j in range(i + 1, len(self.numeric_cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.3:  # Threshold cho correlation cao
                    high_corr_pairs.append(
                        {
                            "var1": self.numeric_cols[i],
                            "var2": self.numeric_cols[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        print("Các cặp biến có correlation cao (>0.3):")
        for pair in high_corr_pairs[:10]:  # Hiển thị top 10
            print(f"  {pair['var1']} - {pair['var2']}: {pair['correlation']:.3f}")

        print("-" * 60)

        # Lag selection cho một số biến chính
        key_vars = (
            ["gdp", "fdi", "trade"]
            if all(v in self.numeric_cols for v in ["gdp", "fdi", "trade"])
            else self.numeric_cols[:3]
        )

        for var in key_vars:
            print(f"\nLag selection cho {var}:")

            lag_criteria = []
            countries = self.df["Country Name"].unique()

            for lag in range(1, max_lags + 1):
                valid_tests = 0
                total_criteria = {"AIC": 0, "BIC": 0}

                for country in countries[:5]:  # Test trên 5 quốc gia đầu
                    country_data = self.df[self.df["Country Name"] == country][
                        var
                    ].dropna()

                    if len(country_data) < 20:
                        continue

                    try:
                        # Tạo lagged variables
                        data_for_reg = pd.DataFrame({"y": country_data})
                        for lag_idx in range(1, lag + 1):
                            data_for_reg[f"y_lag{lag_idx}"] = country_data.shift(
                                lag_idx
                            )

                        data_for_reg = data_for_reg.dropna()

                        if len(data_for_reg) < 10:
                            continue

                        # Fit AR model
                        y = data_for_reg["y"]
                        X = data_for_reg.drop("y", axis=1)

                        if len(X.columns) > 0:
                            model = OLS(y, X).fit()

                            n = len(y)
                            k = len(X.columns)

                            # Calculate AIC and BIC
                            aic = n * np.log(model.ssr / n) + 2 * k
                            bic = n * np.log(model.ssr / n) + k * np.log(n)

                            total_criteria["AIC"] += aic
                            total_criteria["BIC"] += bic
                            valid_tests += 1
                    except Exception:
                        continue

                if valid_tests > 0:
                    avg_aic = total_criteria["AIC"] / valid_tests
                    avg_bic = total_criteria["BIC"] / valid_tests

                    lag_criteria.append(
                        {
                            "lag": lag,
                            "AIC": avg_aic,
                            "BIC": avg_bic,
                            "valid_tests": valid_tests,
                        }
                    )

            if lag_criteria:
                # Tìm lag tối ưu
                best_aic_lag = min(lag_criteria, key=lambda x: x["AIC"])["lag"]
                best_bic_lag = min(lag_criteria, key=lambda x: x["BIC"])["lag"]

                print(f"  Lag tối ưu theo AIC: {best_aic_lag}")
                print(f"  Lag tối ưu theo BIC: {best_bic_lag}")

                # Hiển thị chi tiết
                print("  Chi tiết các lag:")
                for criteria in lag_criteria:
                    print(
                        f"    Lag {criteria['lag']}: AIC={criteria['AIC']:.2f}, BIC={criteria['BIC']:.2f}"
                    )

                self.results["lag_selection"][var] = {
                    "best_aic_lag": best_aic_lag,
                    "best_bic_lag": best_bic_lag,
                    "criteria": lag_criteria,
                }

        print("-" * 60)

    def cointegration_tests(self):
        """
        Kiểm định đồng liên kết (Cointegration)
        """
        print("=== KIỂM ĐỊNH ĐỒNG LIÊN KẾT ===")

        self.results["cointegration"] = {}

        # Chọn các biến chính để test đồng liên kết
        key_vars = []
        if "gdp" in self.numeric_cols:
            key_vars.append("gdp")
        if "fdi" in self.numeric_cols:
            key_vars.append("fdi")
        if "trade" in self.numeric_cols:
            key_vars.append("trade")
        if len(key_vars) < 2:
            key_vars = self.numeric_cols[:3]

        print(f"Kiểm định đồng liên kết cho các biến: {key_vars}")

        # Engle-Granger cointegration test cho từng quốc gia
        countries = self.df["Country Name"].unique()

        for i in range(len(key_vars)):
            for j in range(i + 1, len(key_vars)):
                var1, var2 = key_vars[i], key_vars[j]
                print(f"\n--- Đồng liên kết giữa {var1} và {var2} ---")

                cointegration_results = []

                for country in countries[:10]:  # Test 10 quốc gia đầu
                    country_data = self.df[self.df["Country Name"] == country][
                        [var1, var2]
                    ].dropna()

                    if len(country_data) < 15:
                        continue

                    try:
                        # Engle-Granger cointegration test
                        coint_stat, coint_pval, crit_vals = coint(
                            country_data[var1], country_data[var2]
                        )

                        cointegration_results.append(
                            {
                                "country": country,
                                "statistic": coint_stat,
                                "p_value": coint_pval,
                                "cointegrated": coint_pval < 0.05,
                                "critical_1%": crit_vals[0],
                                "critical_5%": crit_vals[1],
                                "critical_10%": crit_vals[2],
                            }
                        )
                    except Exception:
                        continue

                if cointegration_results:
                    cointegrated_count = sum(
                        [r["cointegrated"] for r in cointegration_results]
                    )
                    cointegrated_pct = (
                        cointegrated_count / len(cointegration_results) * 100
                    )

                    print(
                        f"Kết quả: {cointegrated_count}/{len(cointegration_results)} ({cointegrated_pct:.1f}%) quốc gia có đồng liên kết"
                    )

                    # Mẫu kết quả
                    print("  Mẫu kết quả:")
                    for result in cointegration_results[:3]:
                        status = (
                            "Có đồng liên kết"
                            if result["cointegrated"]
                            else "Không có đồng liên kết"
                        )
                        print(
                            f"    {result['country']}: {result['statistic']:.3f} (p={result['p_value']:.3f}) - {status}"
                        )

                    self.results["cointegration"][f"{var1}_{var2}"] = (
                        cointegration_results
                    )

        print("-" * 60)

    def generate_summary_report(self):
        """
        Tạo báo cáo tổng hợp
        """
        print("=== BÁO CÁO TỔNG HỢP ===")

        # 1. Tóm tắt tính dừng
        print("1. TÓM TẮT TÍNH DỪNG:")
        non_stationary_vars = []
        stationary_vars = []

        for var in self.numeric_cols:
            if var in self.results.get("individual_tests", {}):
                adf_results = self.results["individual_tests"][var].get("adf", [])
                if adf_results:
                    stationary_pct = (
                        sum([r["stationary"] for r in adf_results])
                        / len(adf_results)
                        * 100
                    )
                    if stationary_pct < 50:
                        non_stationary_vars.append(f"{var} ({stationary_pct:.1f}%)")
                    else:
                        stationary_vars.append(f"{var} ({stationary_pct:.1f}%)")

        print(
            f"   - Biến có tính dừng: {', '.join(stationary_vars) if stationary_vars else 'Không có'}"
        )
        print(
            f"   - Biến không dừng: {', '.join(non_stationary_vars) if non_stationary_vars else 'Không có'}"
        )

        # 2. Khuyến nghị sai phân
        print("\n2. KHUYẾN NGHỊ XỬ LÝ:")
        if non_stationary_vars:
            print("   - Các biến không dừng cần sai phân bậc 1")
            print("   - Kiểm tra lại tính dừng sau sai phân")

        if len(non_stationary_vars) >= 2:
            print("   - Có thể tồn tại đồng liên kết giữa các biến không dừng")
            print("   - Xem xét sử dụng mô hình VECM (Vector Error Correction Model)")

        # 3. Cấu trúc độ trễ
        print("\n3. CẤU TRÚC ĐỘ TRỄ:")
        if "lag_selection" in self.results:
            for var, lag_info in self.results["lag_selection"].items():
                print(
                    f"   - {var}: AIC lag={lag_info['best_aic_lag']}, BIC lag={lag_info['best_bic_lag']}"
                )

        # 4. Đồng liên kết
        print("\n4. ĐỒNG LIÊN KẾT:")
        if "cointegration" in self.results:
            for var_pair, results in self.results["cointegration"].items():
                if results:
                    cointegrated_pct = (
                        sum([r["cointegrated"] for r in results]) / len(results) * 100
                    )
                    print(
                        f"   - {var_pair.replace('_', ' & ')}: {cointegrated_pct:.1f}% quốc gia có đồng liên kết"
                    )

        print("\n5. KHUYẾN NGHỊ MÔ HÌNH:")
        if non_stationary_vars and "cointegration" in self.results:
            has_cointegration = any(
                [
                    sum([r["cointegrated"] for r in results]) / len(results) > 0.3
                    for results in self.results["cointegration"].values()
                    if results
                ]
            )

            if has_cointegration:
                print("   - Sử dụng mô hình VECM (Vector Error Correction Model)")
                print("   - Đưa error correction term vào mô hình")
            else:
                print("   - Sử dụng mô hình VAR với các biến đã sai phân")
        else:
            print("   - Sử dụng mô hình VAR với các biến gốc hoặc đã sai phân")

        print("-" * 60)

    def run_full_analysis(self):
        """
        Chạy toàn bộ phân tích
        """
        print("BẮTĐẦU PHÂN TÍCH TÍNH DỪNG VÀ ĐỒNG LIÊN KẾT")
        print("=" * 60)

        # Chuẩn bị dữ liệu
        self.prepare_data()

        # Kiểm định unit root cho từng quốc gia
        self.unit_root_tests_individual()

        # Tóm tắt kết quả panel
        summary_df = self.panel_unit_root_summary()

        # Kiểm định sai phân bậc 1
        self.first_difference_tests()

        # Xác định cấu trúc độ trễ
        self.lag_selection()

        # Kiểm định đồng liên kết
        self.cointegration_tests()

        # Báo cáo tổng hợp
        self.generate_summary_report()

        return self.results, summary_df


# Hàm chính
def main():
    """
    Hàm chính để chạy phân tích
    """
    try:
        # Đọc dữ liệu
        print("Đang đọc dữ liệu...")
        df = pd.read_csv(
            r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\processed_data.csv"
        )  # Thay đổi đường dẫn ở đây

        # Khởi tạo và chạy phân tích
        analyzer = PanelStationarityTests(df)
        results, summary = analyzer.run_full_analysis()

        # Lưu kết quả
        summary.to_csv("stationarity_summary.csv", index=False)
        print("\nBáo cáo tóm tắt đã được lưu: stationarity_summary.csv")

        return results, summary

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu.")
        return None, None
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None, None


# Chạy phần tích
if __name__ == "__main__":
    results, summary = main()
