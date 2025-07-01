import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

warnings.filterwarnings("ignore")


class StationarityAnalysisFollowup:
    """
    Phân tích chi tiết kết quả kiểm định tính dừng và xử lý tiếp theo
    """

    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = [
            col
            for col in df.columns
            if col not in ["Time", "Country Name", "Country Code"]
        ]

        # Kết quả kiểm định từ bảng
        self.stationarity_results = {
            "gdp": {"adf": 5.3, "kpss": 5.3, "consensus": 0.0},
            "hte": {"adf": 15.8, "kpss": 52.6, "consensus": 10.5},
            "ict_exp": {"adf": 21.1, "kpss": 38.6, "consensus": 14.0},
            "fdi": {"adf": 45.6, "kpss": 78.9, "consensus": 35.1},
            "inet_usr": {"adf": 17.5, "kpss": 1.8, "consensus": 0.0},
            "sec_srv": {"adf": 5.3, "kpss": 3.5, "consensus": 0.0},
            "mob_sub": {"adf": 36.8, "kpss": 38.6, "consensus": 17.5},
            "ter_enr": {"adf": 26.3, "kpss": 31.6, "consensus": 12.3},
            "edu_exp": {"adf": 29.8, "kpss": 59.6, "consensus": 21.1},
            "sci_art": {"adf": 15.8, "kpss": 3.5, "consensus": 0.0},
            "pop": {"adf": 22.8, "kpss": 5.3, "consensus": 3.5},
            "infl": {"adf": 21.1, "kpss": 78.9, "consensus": 21.1},
            "trade": {"adf": 36.8, "kpss": 56.1, "consensus": 24.6},
        }

    def analyze_stationarity_patterns(self):
        """
        Phân tích patterns từ kết quả kiểm định
        """
        print("=== PHÂN TÍCH PATTERNS TÍNH DỪNG ===")

        # Phân loại biến theo mức độ dừng
        highly_non_stationary = []  # consensus < 5%
        moderately_non_stationary = []  # consensus 5-20%
        somewhat_stationary = []  # consensus 20-40%

        conflicting_tests = []  # ADF và KPSS khác biệt lớn

        for var, results in self.stationarity_results.items():
            consensus = results["consensus"]
            adf_diff = abs(results["adf"] - results["kpss"])

            if consensus < 5:
                highly_non_stationary.append(var)
            elif consensus < 20:
                moderately_non_stationary.append(var)
            else:
                somewhat_stationary.append(var)

            if adf_diff > 30:  # Khác biệt lớn giữa ADF và KPSS
                conflicting_tests.append(
                    {
                        "var": var,
                        "adf": results["adf"],
                        "kpss": results["kpss"],
                        "difference": adf_diff,
                    }
                )

        print("1. BIẾN CỰC KỲ KHÔNG DỪNG (consensus < 5%):")
        print(f"   {highly_non_stationary}")
        print("   → Chắc chắn cần differencing")

        print("\n2. BIẾN KHÁ KHÔNG DỪNG (consensus 5-20%):")
        print(f"   {moderately_non_stationary}")
        print("   → Nên differencing, có thể có đồng liên kết")

        print("\n3. BIẾN CÓ TÍNH DỪNG PHẦN NÀO (consensus 20-40%):")
        print(f"   {somewhat_stationary}")
        print("   → Cần kiểm tra thêm, có thể dùng levels")

        print("\n4. BIẾN CÓ KẾT QUẢ XUNG ĐỘT:")
        for conflict in conflicting_tests:
            print(
                f"   {conflict['var']}: ADF={conflict['adf']}%, KPSS={conflict['kpss']}% (diff={conflict['difference']:.1f}%)"
            )
        print("   → Cần điều tra thêm về cấu trúc dữ liệu")

        print("-" * 60)

        return {
            "highly_non_stationary": highly_non_stationary,
            "moderately_non_stationary": moderately_non_stationary,
            "somewhat_stationary": somewhat_stationary,
            "conflicting_tests": conflicting_tests,
        }

    def create_differenced_data(self):
        """
        Tạo dữ liệu sai phân và kiểm định lại
        """
        print("=== TẠO DỮ LIỆU SAI PHÂN VÀ KIỂM ĐỊNH LẠI ===")

        # Tạo dữ liệu sai phân
        df_diff = self.df.copy()

        # Thêm cột sai phân cho tất cả biến số
        for var in self.numeric_cols:
            if var in df_diff.columns:
                df_diff[f"d_{var}"] = df_diff.groupby("Country Name")[var].diff()

        print(f"Đã tạo {len(self.numeric_cols)} biến sai phân")

        # Kiểm định tính dừng cho một số biến sai phân mẫu
        sample_vars = (
            ["gdp", "fdi", "trade", "hte"]
            if all(v in self.numeric_cols for v in ["gdp", "fdi", "trade", "hte"])
            else self.numeric_cols[:4]
        )

        diff_stationarity_results = {}

        for var in sample_vars:
            print(f"\n--- Kiểm định tính dừng cho d_{var} ---")

            diff_var = f"d_{var}"
            if diff_var not in df_diff.columns:
                continue

            adf_stationary_count = 0
            kpss_stationary_count = 0
            total_countries = 0

            countries = df_diff["Country Name"].unique()

            for country in countries:
                country_data = df_diff[df_diff["Country Name"] == country][
                    diff_var
                ].dropna()

                if len(country_data) < 10:
                    continue

                total_countries += 1

                try:
                    # ADF test
                    adf_stat, adf_pval, _, _, _, _ = adfuller(
                        country_data, autolag="AIC"
                    )
                    if adf_pval < 0.05:
                        adf_stationary_count += 1

                    # KPSS test
                    kpss_stat, kpss_pval, _, _ = kpss(country_data, regression="c")
                    if kpss_pval > 0.05:
                        kpss_stationary_count += 1

                except Exception:
                    total_countries -= 1
                    continue

            if total_countries > 0:
                adf_pct = (adf_stationary_count / total_countries) * 100
                kpss_pct = (kpss_stationary_count / total_countries) * 100
                consensus_pct = min(adf_pct, kpss_pct)

                diff_stationarity_results[var] = {
                    "adf_pct": adf_pct,
                    "kpss_pct": kpss_pct,
                    "consensus_pct": consensus_pct,
                    "improvement": consensus_pct
                    - self.stationarity_results[var]["consensus"],
                }

                print(
                    f"   ADF dừng: {adf_pct:.1f}% (gốc: {self.stationarity_results[var]['adf']:.1f}%)"
                )
                print(
                    f"   KPSS dừng: {kpss_pct:.1f}% (gốc: {self.stationarity_results[var]['kpss']:.1f}%)"
                )
                print(
                    f"   Consensus: {consensus_pct:.1f}% (gốc: {self.stationarity_results[var]['consensus']:.1f}%)"
                )
                print(
                    f"   Cải thiện: +{consensus_pct - self.stationarity_results[var]['consensus']:.1f}%"
                )

        print("-" * 60)

        return df_diff, diff_stationarity_results

    def cointegration_analysis(self):
        """
        Phân tích đồng liên kết chi tiết
        """
        print("=== PHÂN TÍCH ĐỒNG LIÊN KẾT CHI TIẾT ===")

        # Chọn các biến có khả năng đồng liên kết cao (cùng I(1))
        potential_cointegrated_vars = []
        for var, results in self.stationarity_results.items():
            if results["consensus"] < 30:  # Biến không dừng
                potential_cointegrated_vars.append(var)

        print(f"Biến có khả năng đồng liên kết: {potential_cointegrated_vars}")

        # Thực hiện Johansen cointegration test cho một số quốc gia mẫu
        if len(potential_cointegrated_vars) >= 3:
            test_vars = potential_cointegrated_vars[:4]  # Lấy 4 biến đầu
            countries = self.df["Country Name"].unique()

            cointegration_summary = {}

            for country in countries[:5]:  # Test 5 quốc gia đầu
                print(f"\n--- Johansen Test cho {country} ---")

                country_data = self.df[self.df["Country Name"] == country][
                    test_vars
                ].dropna()

                if len(country_data) < 20:
                    print(f"   Không đủ dữ liệu (chỉ có {len(country_data)} quan sát)")
                    continue

                try:
                    # Johansen cointegration test
                    johansen_result = coint_johansen(
                        country_data, det_order=0, k_ar_diff=1
                    )

                    # Kiểm tra số lượng vector đồng liên kết
                    trace_stats = johansen_result.lr1
                    critical_values = johansen_result.cvt

                    cointegrating_vectors = 0
                    for i in range(len(trace_stats)):
                        if trace_stats[i] > critical_values[i, 1]:  # 5% critical value
                            cointegrating_vectors += 1

                    cointegration_summary[country] = {
                        "cointegrating_vectors": cointegrating_vectors,
                        "trace_stats": trace_stats,
                        "max_stat": johansen_result.lr2[0]
                        if len(johansen_result.lr2) > 0
                        else 0,
                    }

                    print(f"   Số vector đồng liên kết: {cointegrating_vectors}")
                    print(f"   Trace statistic: {trace_stats[0]:.2f}")

                except Exception as e:
                    print(f"   Lỗi trong Johansen test: {str(e)}")
                    continue

            # Tóm tắt kết quả đồng liên kết
            if cointegration_summary:
                avg_vectors = np.mean(
                    [
                        result["cointegrating_vectors"]
                        for result in cointegration_summary.values()
                    ]
                )
                countries_with_coint = sum(
                    [
                        1
                        for result in cointegration_summary.values()
                        if result["cointegrating_vectors"] > 0
                    ]
                )

                print("\nTÓM TẮT ĐỒNG LIÊN KẾT:")
                print(f"   Số vector đồng liên kết trung bình: {avg_vectors:.1f}")
                print(
                    f"   Số quốc gia có đồng liên kết: {countries_with_coint}/{len(cointegration_summary)}"
                )

                return cointegration_summary

        print("-" * 60)
        return {}

    def model_specification_recommendations(self):
        """
        Đưa ra khuyến nghị cụ thể về specification mô hình
        """
        print("=== KHUYẾN NGHỊ SPECIFICATION MÔ HÌNH ===")

        # Phân tích dựa trên kết quả
        highly_non_stationary = [
            var
            for var, res in self.stationarity_results.items()
            if res["consensus"] < 5
        ]
        moderately_non_stationary = [
            var
            for var, res in self.stationarity_results.items()
            if 5 <= res["consensus"] < 20
        ]
        somewhat_stationary = [
            var
            for var, res in self.stationarity_results.items()
            if res["consensus"] >= 20
        ]

        print("1. CHIẾN LƯỢC XỬ LÝ THEO NHÓM:")
        print(f"   Group 1 - Chắc chắn I(1): {highly_non_stationary}")
        print("   → Sử dụng first difference hoặc kiểm tra cointegration")

        print(f"   Group 2 - Có thể I(1): {moderately_non_stationary}")
        print("   → Ưu tiên first difference, nhưng có thể test cả levels")

        print(f"   Group 3 - Có thể I(0): {somewhat_stationary}")
        print("   → Có thể sử dụng levels, nhưng cần kiểm tra robustness")

        print("\n2. CÁC PHƯƠNG ÁN MÔ HÌNH:")

        print("\n   PHƯƠNG ÁN 1: TOÀN BỘ FIRST DIFFERENCE")
        print("   - Ưu điểm: An toàn, tránh spurious regression")
        print("   - Nhược điểm: Mất thông tin long-run relationship")
        print("   - Phù hợp: Khi không có đồng liên kết rõ ràng")

        print("\n   PHƯƠNG ÁN 2: VECTOR ERROR CORRECTION MODEL (VECM)")
        print("   - Điều kiện: Có ít nhất 2 biến I(1) và có đồng liên kết")
        print("   - Ưu điểm: Giữ được both short-run và long-run relationships")
        print("   - Phù hợp: Khi có bằng chứng đồng liên kết mạnh")

        print("\n   PHƯƠNG ÁN 3: MIXED APPROACH")
        print("   - I(1) variables: Sử dụng first difference")
        print("   - I(0) variables: Sử dụng levels")
        print("   - Cần: Kiểm tra cẩn thận về spurious correlation")

        print("\n3. KIỂM ĐỊNH BỔ SUNG CẦN THỰC HIỆN:")
        print("   ✓ Kiểm định unit root cho first differences")
        print("   ✓ Kiểm định đồng liên kết (Johansen/Engle-Granger)")
        print("   ✓ Lag selection cho VAR/VECM")
        print("   ✓ Residual diagnostics")
        print("   ✓ Structural break tests")

        print("\n4. KHUYẾN NGHỊ CỤ THỂ CHO DATASET:")

        total_non_stationary = len(highly_non_stationary) + len(
            moderately_non_stationary
        )

        if total_non_stationary >= 8:  # Majority non-stationary
            print("   → KHUYẾN NGHỊ CHÍNH: First Difference VAR")
            print("     * Lý do: Majority biến không dừng")
            print("     * Bước 1: Transform tất cả biến thành first difference")
            print("     * Bước 2: Kiểm định tính dừng của differences")
            print("     * Bước 3: Estimate VAR với differenced data")

        if len(highly_non_stationary) >= 3:
            print("   → KHUYẾN NGHỊ PHỤ: Kiểm tra VECM")
            print("     * Lý do: Nhiều biến I(1) có thể có đồng liên kết")
            print("     * Test Johansen cointegration")
            print("     * Nếu có cointegration → VECM")
            print("     * Nếu không → VAR với differences")

        print("\n5. IMPLEMENTATION STEPS:")
        print("   Step 1: Tạo first differences cho tất cả biến")
        print("   Step 2: Kiểm định unit root cho differences")
        print("   Step 3: Lag selection (AIC/BIC/HQ)")
        print("   Step 4: Johansen cointegration test")
        print("   Step 5: Estimate VECM hoặc VAR")
        print("   Step 6: Diagnostic tests")
        print("   Step 7: Impulse response analysis")

        print("-" * 60)

    def generate_code_template(self):
        """
        Tạo template code cho bước tiếp theo
        """
        print("=== TEMPLATE CODE CHO BƯỚC TIẾP THEO ===")

        code_template = """
# BƯỚC 1: TẠO FIRST DIFFERENCES
def create_differences(df):
    df_diff = df.copy()
    numeric_cols = ['gdp', 'hte', 'ict_exp', 'fdi', 'inet_usr', 'sec_srv', 
                   'mob_sub', 'ter_enr', 'edu_exp', 'sci_art', 'pop', 'infl', 'trade']
    
    for var in numeric_cols:
        df_diff[f'd_{var}'] = df_diff.groupby('Country Name')[var].diff()
    
    return df_diff

# BƯỚC 2: KIỂM ĐỊNH UNIT ROOT CHO DIFFERENCES
from statsmodels.tsa.stattools import adfuller

def test_difference_stationarity(df_diff):
    diff_vars = [col for col in df_diff.columns if col.startswith('d_')]
    results = {}
    
    for var in diff_vars:
        stationary_countries = 0
        total_countries = 0
        
        for country in df_diff['Country Name'].unique():
            country_data = df_diff[df_diff['Country Name'] == country][var].dropna()
            if len(country_data) >= 10:
                total_countries += 1
                adf_stat, adf_pval, _, _, _, _ = adfuller(country_data)
                if adf_pval < 0.05:
                    stationary_countries += 1
        
        if total_countries > 0:
            results[var] = stationary_countries / total_countries * 100
    
    return results

# BƯỚC 3: JOHANSEN COINTEGRATION TEST
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(df, variables, country):
    country_data = df[df['Country Name'] == country][variables].dropna()
    
    if len(country_data) < 20:
        return None
    
    try:
        result = coint_johansen(country_data, det_order=0, k_ar_diff=1)
        
        # Số lượng cointegrating vectors
        trace_stats = result.lr1
        critical_values = result.cvt
        
        cointegrating_vectors = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > critical_values[i, 1]:  # 5% level
                cointegrating_vectors += 1
        
        return cointegrating_vectors
    except:
        return None

# BƯỚC 4: LAG SELECTION
from statsmodels.tsa.vector_ar.var_model import VAR

def select_lag_order(data, maxlags=8):
    model = VAR(data)
    lag_order = model.select_order(maxlags=maxlags)
    return lag_order

# BƯỚC 5: ESTIMATE VECM hoặc VAR
from statsmodels.tsa.vector_ar.vecm import VECM

def estimate_vecm(data, coint_rank, k_ar_diff=1):
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
    vecm_result = model.fit()
    return vecm_result

def estimate_var(data, lags):
    model = VAR(data)
    var_result = model.fit(lags)
    return var_result

# BƯỚC 6: DIAGNOSTIC TESTS
def diagnostic_tests(model_result):
    # Serial correlation test
    serial_corr = model_result.test_serial_correlation(lags=1)
    
    # Normality test
    normality = model_result.test_normality()
    
    # Heteroscedasticity test
    hetero = model_result.test_heteroscedasticity(lags=1)
    
    return {
        'serial_correlation': serial_corr,
        'normality': normality,
        'heteroscedasticity': hetero
    }

# MAIN WORKFLOW
def main_analysis(df):
    # 1. Create differences
    df_diff = create_differences(df)
    
    # 2. Test stationarity of differences
    diff_results = test_difference_stationarity(df_diff)
    print("Stationarity of differences:", diff_results)
    
    # 3. Test cointegration for key countries
    key_vars = ['gdp', 'fdi', 'trade', 'hte']  # Adjust as needed
    
    for country in df['Country Name'].unique()[:5]:  # Test first 5 countries
        coint_rank = johansen_test(df, key_vars, country)
        print(f"{country}: {coint_rank} cointegrating vectors")
        
        if coint_rank and coint_rank > 0:
            # Use VECM
            country_data = df[df['Country Name'] == country][key_vars].dropna()
            if len(country_data) >= 20:
                vecm_result = estimate_vecm(country_data, coint_rank)
                print(f"VECM for {country}:")
                print(vecm_result.summary())
        else:
            # Use VAR with differences
            country_data = df_diff[df_diff['Country Name'] == country]
            diff_vars = [f'd_{var}' for var in key_vars]
            country_data_diff = country_data[diff_vars].dropna()
            
            if len(country_data_diff) >= 20:
                lag_order = select_lag_order(country_data_diff)
                optimal_lag = lag_order.aic  # or bic, hqic
                
                var_result = estimate_var(country_data_diff, optimal_lag)
                print(f"VAR for {country}:")
                print(var_result.summary())
                
                # Diagnostic tests
                diagnostics = diagnostic_tests(var_result)
                print("Diagnostics:", diagnostics)
        """

        print(code_template)
        print("-" * 60)

    def run_complete_analysis(self):
        """
        Chạy toàn bộ phân tích follow-up
        """
        print("PHÂN TÍCH FOLLOW-UP KẾT QUẢ KIỂM ĐỊNH TÍNH DỪNG")
        print("=" * 60)

        # 1. Phân tích patterns
        patterns = self.analyze_stationarity_patterns()

        # 2. Tạo và kiểm định dữ liệu sai phân
        df_diff, diff_results = self.create_differenced_data()

        # 3. Phân tích đồng liên kết
        coint_results = self.cointegration_analysis()

        # 4. Khuyến nghị mô hình
        self.model_specification_recommendations()

        # 5. Template code
        self.generate_code_template()

        return {
            "patterns": patterns,
            "differenced_data": df_diff,
            "difference_stationarity": diff_results,
            "cointegration": coint_results,
        }


# Hàm main
def main():
    """
    Chạy phân tích với dữ liệu thực tế
    """
    try:
        # Đọc dữ liệu
        df = pd.read_csv(
            r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\processed_data.csv"
        )  # Thay đổi đường dẫn

        # Chạy phân tích
        analyzer = StationarityAnalysisFollowup(df)
        results = analyzer.run_complete_analysis()

        return results

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu.")
        return None
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
