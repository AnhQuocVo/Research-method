import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
import warnings

warnings.filterwarnings("ignore")

# Thiết lập style cho plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ICTIntellectualCapitalAnalysis:
    def __init__(self, file_path):
        """
        Khởi tạo class phân tích
        """
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.clusters_df = None
        self.model_results = {}

    def load_and_explore_data(self):
        """
        1. Load và khám phá dữ liệu ban đầu
        """
        print("=" * 60)
        print("1. LOADING VÀ KHÁM PHÁ DỮ LIỆU")
        print("=" * 60)

        # Load data
        if self.file_path.endswith(".csv"):
            self.df = pd.read_csv(
                r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Data\ML Process\Data_csv\rawdata_1.csv"
            )
        else:
            self.df = pd.read_excel(self.file_path)

        print(f"Kích thước dataset: {self.df.shape}")
        print(f"Số quốc gia: {self.df['Country Name'].nunique()}")
        print(f"Thời gian: {self.df['Time'].min()} - {self.df['Time'].max()}")

        # Kiểm tra missing values
        print("\nMissing values theo biến:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {"Missing Count": missing_data, "Percentage": missing_percent}
        ).sort_values("Percentage", ascending=False)
        print(missing_df[missing_df["Missing Count"] > 0])

        # Thống kê mô tả
        print("\nThống kê mô tả các biến chính:")
        key_vars = [
            "gdp",
            "inet_usr",
            "sec_srv",
            "mob_sub",
            "ter_enr",
            "edu_exp",
            "rnd_exp",
            "sci_art",
            "hci",
            "pop",
            "infl",
            "urb_area",
            "trade",
        ]
        print(self.df[key_vars].describe())

        return self.df

    def process_data(self):
        """
        2. Xử lý và chuẩn bị dữ liệu
        """
        print("\n\n" + "=" * 60)
        print("2. XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU")
        print("=" * 60)

        # Tạo bản copy để xử lý
        df_work = self.df.copy()

        # Định nghĩa các biến cần thiết cho mô hình
        model_vars = [
            "Time",
            "Country Name",
            "Country Code",
            "gdp",
            "inet_usr",
            "sec_srv",
            "mob_sub",
            "ter_enr",  # ICT variables
            "edu_exp",
            "rnd_exp",
            "sci_art",
            "hci",  # IC variables
            "pop",
            "infl",
            "urb_area",
            "trade",
        ]  # Control variables

        # Giữ chỉ các biến cần thiết
        df_work = df_work[model_vars]

        print(f"Dữ liệu trước khi drop missing: {df_work.shape}")

        # Drop tất cả rows có missing values
        df_work = df_work.dropna()
        print(f"Dữ liệu sau khi drop missing: {df_work.shape}")

        # Log transformation cho các biến phù hợp (giá trị > 0)
        log_vars = ["gdp", "inet_usr", "sec_srv", "ter_enr", "pop", "trade"]

        for var in log_vars:
            if var in df_work.columns:
                # Chỉ log transform các giá trị > 0
                mask = df_work[var] > 0
                df_work.loc[mask, f"ln_{var}"] = np.log(df_work.loc[mask, var])
                print(f"Log transformed {var}: {mask.sum()} observations")

        # Tạo ICT Index (chuẩn hóa rồi tính trung bình)
        ict_vars = ["ln_inet_usr", "ln_sec_srv", "mob_sub", "ln_ter_enr"]

        # Chuẩn hóa các biến ICT
        scaler = StandardScaler()
        ict_data = df_work[ict_vars].copy()
        ict_data = ict_data.dropna()

        if len(ict_data) > 0:
            ict_scaled = scaler.fit_transform(ict_data)
            df_work.loc[ict_data.index, "ict_index"] = np.mean(ict_scaled, axis=1)
            print(f"Tạo ICT Index thành công cho {len(ict_data)} observations")

        # Tính GDP per capita
        df_work["gdp_pc"] = df_work["gdp"] / df_work["pop"]
        df_work["ln_gdp_pc"] = np.log(df_work["gdp_pc"])

        # Drop những dòng vẫn còn missing sau khi tạo biến mới
        df_work = df_work.dropna()

        print(f"Dữ liệu cuối cùng: {df_work.shape}")
        print(f"Thời gian: {df_work['Time'].min()} - {df_work['Time'].max()}")
        print(f"Số quốc gia: {df_work['Country Name'].nunique()}")

        self.df_processed = df_work
        return df_work

    def cluster_analysis(self):
        """
        3. Phân nhóm quốc gia bằng clustering
        """
        print("\n\n" + "=" * 60)
        print("3. PHÂN NHÓM QUỐC GIA (CLUSTERING)")
        print("=" * 60)

        # Chuẩn bị dữ liệu cho clustering (lấy giá trị trung bình theo quốc gia)
        cluster_vars = ["gdp_pc", "hci", "ict_index"]
        cluster_data = (
            self.df_processed.groupby("Country Name")[cluster_vars].mean().reset_index()
        )
        cluster_data = cluster_data.dropna()

        print(f"Số quốc gia để clustering: {len(cluster_data)}")

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data[cluster_vars])

        # Tìm số cluster tối ưu bằng silhouette score
        silhouette_scores = []
        K_range = range(2, min(8, len(cluster_data) // 2))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        # Chọn số cluster tối ưu
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(
            f"Số cluster tối ưu: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})"
        )

        # Thực hiện clustering với K tối ưu
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_data["Cluster"] = kmeans_final.fit_predict(X_scaled)

        # Phân tích đặc điểm các cluster
        print("\nĐặc điểm các cluster:")
        cluster_summary = cluster_data.groupby("Cluster")[cluster_vars].agg(
            ["mean", "std", "count"]
        )
        print(cluster_summary)

        # Hiển thị một số quốc gia tiêu biểu trong mỗi cluster
        print("\nMột số quốc gia tiêu biểu trong mỗi cluster:")
        for cluster_id in sorted(cluster_data["Cluster"].unique()):
            countries = cluster_data[cluster_data["Cluster"] == cluster_id][
                "Country Name"
            ].tolist()
            print(
                f"Cluster {cluster_id}: {', '.join(countries[:5])}{'...' if len(countries) > 5 else ''}"
            )

        self.clusters_df = cluster_data

        # Merge cluster info back to main dataset
        self.df_processed = self.df_processed.merge(
            cluster_data[["Country Name", "Cluster"]], on="Country Name", how="left"
        )

        return cluster_data

    def run_econometric_models(self):
        """
        4. Chạy các mô hình kinh tế lượng
        """
        print("\n\n" + "=" * 60)
        print("4. MÔ HÌNH KINH TẾ LƯỢNG")
        print("=" * 60)

        # Chuẩn bị dữ liệu cho panel regression
        df_panel = self.df_processed.copy()
        df_panel = df_panel.set_index(["Country Name", "Time"])

        # Định nghĩa biến phụ thuộc và biến độc lập
        y_var = "ln_gdp"

        # Biến độc lập
        X_vars = [
            "ln_inet_usr",
            "ln_sec_srv",
            "mob_sub",
            "ln_ter_enr",  # ICT vars
            "edu_exp",
            "rnd_exp",
            "sci_art",
            "hci",  # IC vars
            "ln_pop",
            "infl",
            "urb_area",
            "ln_trade",  # Control vars
        ]

        # Lọc các biến thực sự có trong data
        available_X_vars = [
            var
            for var in X_vars
            if var in df_panel.columns and not df_panel[var].isnull().all()
        ]
        print(f"Biến độc lập sử dụng: {available_X_vars}")

        df_model = df_panel[[y_var] + available_X_vars].dropna()
        print(f"Số observations cho mô hình: {len(df_model)}")

        if len(df_model) == 0:
            print("Không có dữ liệu để chạy mô hình!")
            return None

        # 1. Pooled OLS
        print("\n--- Pooled OLS ---")
        try:
            pooled_model = PooledOLS(df_model[y_var], df_model[available_X_vars])
            pooled_results = pooled_model.fit(cov_type="robust")
            print(pooled_results.summary)
            self.model_results["pooled"] = pooled_results
        except Exception as e:
            print(f"Lỗi Pooled OLS: {e}")

        # 2. Fixed Effects
        print("\n--- Fixed Effects ---")
        try:
            fe_model = PanelOLS(
                df_model[y_var], df_model[available_X_vars], entity_effects=True
            )
            fe_results = fe_model.fit(cov_type="robust")
            print(fe_results.summary)
            self.model_results["fixed_effects"] = fe_results
        except Exception as e:
            print(f"Lỗi Fixed Effects: {e}")

        # 3. Random Effects
        print("\n--- Random Effects ---")
        try:
            re_model = RandomEffects(df_model[y_var], df_model[available_X_vars])
            re_results = re_model.fit(cov_type="robust")
            print(re_results.summary)
            self.model_results["random_effects"] = re_results
        except Exception as e:
            print(f"Lỗi Random Effects: {e}")

        # So sánh R-squared
        print("\n--- So sánh mô hình ---")
        for model_name, results in self.model_results.items():
            try:
                r2 = (
                    results.rsquared
                    if hasattr(results, "rsquared")
                    else results.rsquared_overall
                )
                print(f"{model_name}: R² = {r2:.4f}")
            except Exception:
                print(f"{model_name}: Không tính được R²")

        return self.model_results

    def create_visualizations(self):
        """
        5. Tạo trực quan hóa
        """
        print("\n\n" + "=" * 60)
        print("5. TRỰC QUAN HÓA")
        print("=" * 60)

        # Setup figure
        plt.figure(figsize=(20, 15))

        # 1. Heatmap correlation matrix
        plt.subplot(2, 3, 1)
        corr_vars = [
            "ln_gdp",
            "ln_inet_usr",
            "ln_sec_srv",
            "mob_sub",
            "ln_ter_enr",
            "edu_exp",
            "rnd_exp",
            "sci_art",
            "hci",
            "ln_pop",
            "infl",
            "urb_area",
            "ln_trade",
        ]
        available_corr_vars = [
            var for var in corr_vars if var in self.df_processed.columns
        ]

        corr_matrix = self.df_processed[available_corr_vars].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Ma trận tương quan các biến", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 2. Cluster visualization (nếu có đủ data)
        if self.clusters_df is not None and len(self.clusters_df) > 0:
            plt.subplot(2, 3, 2)
            scatter = plt.scatter(
                self.clusters_df["gdp_pc"],
                self.clusters_df["hci"],
                c=self.clusters_df["Cluster"],
                cmap="viridis",
                s=100,
                alpha=0.7,
            )
            plt.colorbar(scatter)
            plt.xlabel("GDP per capita", fontweight="bold")
            plt.ylabel("Human Capital Index", fontweight="bold")
            plt.title("Phân cụm quốc gia\n(GDP per capita vs HCI)", fontweight="bold")
            plt.grid(True, alpha=0.3)

        # 3. GDP distribution by cluster
        if "Cluster" in self.df_processed.columns:
            plt.subplot(2, 3, 3)
            for cluster in sorted(self.df_processed["Cluster"].unique()):
                cluster_data = self.df_processed[
                    self.df_processed["Cluster"] == cluster
                ]
                plt.hist(
                    cluster_data["ln_gdp"],
                    alpha=0.6,
                    label=f"Cluster {cluster}",
                    bins=20,
                )
            plt.xlabel("ln(GDP)", fontweight="bold")
            plt.ylabel("Frequency", fontweight="bold")
            plt.title("Phân bố GDP theo cluster", fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 4. ICT Index vs GDP
        plt.subplot(2, 3, 4)
        if "ict_index" in self.df_processed.columns:
            plt.scatter(
                self.df_processed["ict_index"],
                self.df_processed["ln_gdp"],
                alpha=0.6,
                color="blue",
            )
            plt.xlabel("ICT Index", fontweight="bold")
            plt.ylabel("ln(GDP)", fontweight="bold")
            plt.title("ICT Index vs GDP", fontweight="bold")
            plt.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(
                self.df_processed["ict_index"].dropna(),
                self.df_processed.loc[
                    self.df_processed["ict_index"].dropna().index, "ln_gdp"
                ],
                1,
            )
            p = np.poly1d(z)
            plt.plot(
                sorted(self.df_processed["ict_index"].dropna()),
                p(sorted(self.df_processed["ict_index"].dropna())),
                "r--",
                alpha=0.8,
            )

        # 5. Time series của một số quốc gia
        plt.subplot(2, 3, 5)
        top_countries = (
            self.df_processed.groupby("Country Name")["gdp"].mean().nlargest(5).index
        )
        for country in top_countries:
            country_data = self.df_processed[
                self.df_processed["Country Name"] == country
            ]
            if len(country_data) > 1:
                plt.plot(
                    country_data["Time"],
                    country_data["ln_gdp"],
                    marker="o",
                    label=country,
                    linewidth=2,
                )
        plt.xlabel("Year", fontweight="bold")
        plt.ylabel("ln(GDP)", fontweight="bold")
        plt.title("Xu hướng GDP của các quốc gia hàng đầu", fontweight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # 6. Human Capital vs R&D
        plt.subplot(2, 3, 6)
        plt.scatter(
            self.df_processed["hci"],
            self.df_processed["rnd_exp"],
            alpha=0.6,
            color="green",
        )
        plt.xlabel("Human Capital Index", fontweight="bold")
        plt.ylabel("R&D Expenditure (% GDP)", fontweight="bold")
        plt.title("Human Capital vs R&D Investment", fontweight="bold")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("Các biểu đồ đã được tạo thành công!")

    def machine_learning_analysis(self):
        """
        6. Phân tích Machine Learning
        """
        print("\n\n" + "=" * 60)
        print("6. PHÂN TÍCH MACHINE LEARNING")
        print("=" * 60)

        # Chuẩn bị dữ liệu
        ml_vars = [
            "ln_inet_usr",
            "ln_sec_srv",
            "mob_sub",
            "ln_ter_enr",
            "edu_exp",
            "rnd_exp",
            "sci_art",
            "hci",
            "ln_pop",
            "infl",
            "urb_area",
            "ln_trade",
        ]

        available_ml_vars = [var for var in ml_vars if var in self.df_processed.columns]

        df_ml = self.df_processed[["ln_gdp"] + available_ml_vars].dropna()

        if len(df_ml) == 0:
            print("Không có đủ dữ liệu cho ML analysis!")
            return None

        X = df_ml[available_ml_vars]
        y = df_ml["ln_gdp"]

        # Chia data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Random Forest
        print("\n--- Random Forest ---")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        print(f"Random Forest R²: {rf_r2:.4f}")
        print(f"Random Forest RMSE: {rf_rmse:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"Variable": available_ml_vars, "Importance": rf_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\nFeature Importance (Random Forest):")
        print(feature_importance)

        # Gradient Boosting
        print("\n--- Gradient Boosting ---")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)

        gb_pred = gb_model.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

        print(f"Gradient Boosting R²: {gb_r2:.4f}")
        print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")

        # Feature importance cho GB
        gb_feature_importance = pd.DataFrame(
            {"Variable": available_ml_vars, "Importance": gb_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\nFeature Importance (Gradient Boosting):")
        print(gb_feature_importance)

        # Visualize feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Random Forest
        ax1.barh(
            feature_importance["Variable"][:10], feature_importance["Importance"][:10]
        )
        ax1.set_title("Random Forest Feature Importance", fontweight="bold")
        ax1.set_xlabel("Importance", fontweight="bold")

        # Gradient Boosting
        ax2.barh(
            gb_feature_importance["Variable"][:10],
            gb_feature_importance["Importance"][:10],
        )
        ax2.set_title("Gradient Boosting Feature Importance", fontweight="bold")
        ax2.set_xlabel("Importance", fontweight="bold")

        plt.tight_layout()
        plt.show()

        return {
            "random_forest": {
                "model": rf_model,
                "r2": rf_r2,
                "rmse": rf_rmse,
                "importance": feature_importance,
            },
            "gradient_boosting": {
                "model": gb_model,
                "r2": gb_r2,
                "rmse": gb_rmse,
                "importance": gb_feature_importance,
            },
        }

    def generate_final_report(self):
        """
        7. Tạo báo cáo tổng hợp và gợi ý
        """
        print("\n\n" + "=" * 80)
        print("7. BÁO CÁO TỔNG HỢP VÀ GỢI Ý")
        print("=" * 80)

        print("\n🔍 TỔNG QUAN NGHIÊN CứU:")
        print("─" * 50)
        print(f"• Dataset cuối cùng: {self.df_processed.shape}")
        print(f"• Số quốc gia: {self.df_processed['Country Name'].nunique()}")
        print(
            f"• Khoảng thời gian: {self.df_processed['Time'].min()} - {self.df_processed['Time'].max()}"
        )

        if self.clusters_df is not None:
            print(f"• Số cluster quốc gia: {self.clusters_df['Cluster'].nunique()}")

        print("\n📊 KẾT QUẢ MÔ HÌNH KINH TẾ LƯỢNG:")
        print("─" * 50)

        if self.model_results:
            best_model = None
            best_r2 = -1

            for model_name, results in self.model_results.items():
                try:
                    r2 = (
                        results.rsquared
                        if hasattr(results, "rsquared")
                        else results.rsquared_overall
                    )
                    print(f"• {model_name.replace('_', ' ').title()}: R² = {r2:.4f}")

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
                except Exception:
                    print(
                        f"• {model_name.replace('_', ' ').title()}: Không tính được R²"
                    )

            if best_model:
                print(
                    f"\n✅ MÔ HÌNH ĐƯỢC ĐỀ XUẤT: {best_model.replace('_', ' ').title()}"
                )
                print(f"   R² = {best_r2:.4f}")

        print("\n💡 GIẢI THÍCH Ý NGHĨA CÁC BIẾN:")
        print("─" * 50)
        print("📱 ICT Variables:")
        print("   • ln_inet_usr: Internet users - phản ánh khả năng tiếp cận thông tin")
        print("   • ln_sec_srv: Secure internet servers - đo lường hạ tầng ICT an toàn")
        print("   • mob_sub: Mobile subscriptions - mức độ phổ biến công nghệ di động")
        print("   • ln_ter_enr: Tertiary enrollment - giáo dục đại học, nền tảng ICT")

        print("\n🧠 Intellectual Capital Variables:")
        print("   • edu_exp: Education expenditure - đầu tư vào vốn nhân lực")
        print("   • rnd_exp: R&D expenditure - đầu tư nghiên cứu phát triển")
        print("   • sci_art: Scientific articles - đầu ra nghiên cứu khoa học")
        print("   • hci: Human capital index - chỉ số vốn nhân lực tổng hợp")

        print("\n🎯 GỢI Ý CHÍNH SÁCH:")
        print("─" * 50)
        print("1. 🚀 Đầu tư ICT: Phát triển hạ tầng internet và công nghệ số")
        print("2. 📚 Giáo dục: Tăng chi tiêu giáo dục, đặc biệt giáo dục đại học")
        print("3. 🔬 R&D: Khuyến khích đầu tư nghiên cứu phát triển")
        print("4. 👥 Vốn nhân lực: Phát triển kỹ năng số và năng lực sáng tạo")

        print("\n📈 GỢI Ý CẢI THIỆN NGHIÊN CỨU:")
        print("─" * 50)
        print("1. 📊 Dữ liệu: Thu thập thêm dữ liệu về:")
        print("   • Chất lượng ICT (tốc độ internet, độ tin cậy)")
        print("   • Innovation metrics (patents, startups)")
        print("   • Digital skills của lao động")

        print("2. 🔧 Mô hình: Xem xét:")
        print("   • Nonlinear relationships")
        print("   • Interaction effects giữa ICT và IC")
        print("   • Dynamic panel models")
        print("   • Threshold effects")

        print("3. 🌍 Phân tích: Mở rộng:")
        print("   • So sánh theo nhóm quốc gia (developed vs developing)")
        print("   • Phân tích theo ngành kinh tế")
        print("   • Time-varying coefficients")

        print("\n" + "=" * 80)
        print("✅ PHÂN TÍCH HOÀN TẤT!")
        print("=" * 80)


def main():
    """
    Hàm main để chạy toàn bộ phân tích
    """
    # Thay đổi đường dẫn file của bạn ở đây
    file_path = "your_dataset.xlsx"  # hoặc "your_dataset.csv"

    print("🚀 BẮT ĐẦU PHÂN TÍCH ICT VÀ INTELLECTUAL CAPITAL")
    print("=" * 80)

    # Khởi tạo analyzer
    analyzer = ICTIntellectualCapitalAnalysis(file_path)

    try:
        # Chạy từng bước phân tích
        analyzer.load_and_explore_data()
        analyzer.process_data()
        analyzer.cluster_analysis()
        analyzer.run_econometric_models()
        analyzer.create_visualizations()
        analyzer.machine_learning_analysis()
        analyzer.generate_final_report()

        print("\n🎉 Phân tích hoàn tất thành công!")

        return analyzer

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file dữ liệu!")
        print("   Vui lòng kiểm tra đường dẫn file trong biến 'file_path'")
        return None
    except Exception as e:
        print(f"❌ Lỗi trong quá trình phân tích: {str(e)}")
        print("   Vui lòng kiểm tra dữ liệu và thử lại")
        return None


# Hàm tiện ích để chạy phân tích nhanh
def quick_analysis(file_path, save_results=False):
    """
    Chạy phân tích nhanh với các tùy chọn cơ bản

    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file dữ liệu
    save_results : bool
        Có lưu kết quả không (default: False)

    Returns:
    --------
    analyzer : ICTIntellectualCapitalAnalysis
        Object chứa kết quả phân tích
    """
    analyzer = ICTIntellectualCapitalAnalysis(file_path)

    try:
        print("🔄 Đang chạy phân tích nhanh...")

        # Load và process data
        analyzer.load_and_explore_data()
        analyzer.process_data()

        # Clustering
        analyzer.cluster_analysis()

        # Econometric models
        analyzer.run_econometric_models()

        # Visualizations
        analyzer.create_visualizations()

        # ML analysis
        analyzer.machine_learning_analysis()

        # Final report
        analyzer.generate_final_report()

        # Lưu kết quả nếu được yêu cầu
        if save_results:
            save_analysis_results(analyzer)

        return analyzer

    except Exception as e:
        print(f"❌ Lỗi trong quick_analysis: {str(e)}")
        return None


def save_analysis_results(analyzer, output_dir="analysis_results"):
    """
    Lưu kết quả phân tích ra file

    Parameters:
    -----------
    analyzer : ICTIntellectualCapitalAnalysis
        Object chứa kết quả phân tích
    output_dir : str
        Thư mục lưu kết quả
    """
    import os

    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Lưu processed data
        if analyzer.df_processed is not None:
            analyzer.df_processed.to_csv(
                f"{output_dir}/processed_data.csv", index=False
            )
            print(f"✅ Đã lưu processed data: {output_dir}/processed_data.csv")

        # Lưu cluster results
        if analyzer.clusters_df is not None:
            analyzer.clusters_df.to_csv(
                f"{output_dir}/cluster_results.csv", index=False
            )
            print(f"✅ Đã lưu cluster results: {output_dir}/cluster_results.csv")

        # Lưu model results summary
        if analyzer.model_results:
            with open(f"{output_dir}/model_summary.txt", "w", encoding="utf-8") as f:
                f.write("SUMMARY OF ECONOMETRIC MODELS\n")
                f.write("=" * 50 + "\n")

                for model_name, results in analyzer.model_results.items():
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    try:
                        f.write(str(results.summary))
                        f.write("\n" + "=" * 50 + "\n")
                    except Exception as e:
                        f.write(f"Could not save model summary: {e}\n")

            print(f"✅ Đã lưu model summary: {output_dir}/model_summary.txt")

        print(f"📁 Tất cả kết quả đã được lưu trong thư mục: {output_dir}/")

    except Exception as e:
        print(f"❌ Lỗi khi lưu kết quả: {str(e)}")


# Hàm để tạo data mẫu (nếu cần test)
def create_sample_data(n_countries=20, n_years=10, filename="sample_ict_data.csv"):
    """
    Tạo dữ liệu mẫu để test code

    Parameters:
    -----------
    n_countries : int
        Số lượng quốc gia
    n_years : int
        Số năm
    filename : str
        Tên file output
    """
    np.random.seed(42)

    countries = [f"Country_{i + 1}" for i in range(n_countries)]
    years = list(range(2014, 2014 + n_years))

    data = []

    for country in countries:
        base_gdp = np.random.uniform(1e10, 1e12)  # Base GDP
        base_pop = np.random.uniform(1e6, 1e8)  # Base population

        for year in years:
            # Tăng trưởng theo thời gian
            growth_factor = (year - 2014) * 0.02 + np.random.normal(0, 0.05)

            row = {
                "Time": year,
                "Country Name": country,
                "Country Code": country[:3].upper(),
                "gdp": base_gdp * (1 + growth_factor),
                "hte": np.random.uniform(0.5, 0.9),
                "ict_exp": np.random.uniform(2, 8),
                "fdi": np.random.uniform(-5, 15),
                "inet_usr": np.random.uniform(20, 95),
                "sec_srv": np.random.uniform(1, 100),
                "mob_sub": np.random.uniform(50, 150),
                "ter_enr": np.random.uniform(10, 80),
                "edu_exp": np.random.uniform(3, 7),
                "rnd_exp": np.random.uniform(0.5, 4),
                "sci_art": np.random.uniform(100, 5000),
                "hci": np.random.uniform(0.4, 0.8),
                "pop": base_pop * (1 + growth_factor * 0.5),
                "infl": np.random.uniform(-2, 10),
                "urb_area": np.random.uniform(30, 95),
                "trade": np.random.uniform(20, 200),
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Đã tạo dữ liệu mẫu: {filename}")
    print(f"   Kích thước: {df.shape}")
    print(f"   Quốc gia: {n_countries}, Năm: {n_years}")

    return df


# Chạy phân tích chính
if __name__ == "__main__":
    # HƯỚNG DẪN SỬ DỤNG:
    # 1. Thay đổi đường dẫn file dữ liệu của bạn
    # 2. Chạy script này

    # Tùy chọn 1: Sử dụng dữ liệu thật
    # file_path = "path/to/your/data.xlsx"  # Thay bằng đường dẫn thật
    # analyzer = main()

    # Tùy chọn 2: Tạo và sử dụng dữ liệu mẫu để test
    print("🔧 Tạo dữ liệu mẫu để test...")
    create_sample_data(n_countries=25, n_years=8, filename="sample_ict_data.csv")

    print("\n🚀 Chạy phân tích với dữ liệu mẫu...")
    analyzer = main()

    # Tùy chọn 3: Sử dụng quick_analysis
    # analyzer = quick_analysis("sample_ict_data.csv", save_results=True)

print("""
📋 HƯỚNG DẪN SỬ DỤNG:

1. Cài đặt các thư viện cần thiết:
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels linearmodels

2. Chuẩn bị dữ liệu:
   - File Excel hoặc CSV với các cột theo header bạn cung cấp
   - Đảm bảo có đủ dữ liệu cho phân tích

3. Chạy phân tích:
   - Thay đổi đường dẫn file trong biến file_path
   - Chạy script: python your_script_name.py

4. Các hàm tiện ích:
   - main(): Chạy phân tích đầy đủ
   - quick_analysis(): Chạy phân tích nhanh
   - create_sample_data(): Tạo dữ liệu mẫu
   - save_analysis_results(): Lưu kết quả

5. Kết quả sẽ bao gồm:
   - Thống kê mô tả và khám phá dữ liệu
   - Phân nhóm quốc gia (clustering)
   - Kết quả các mô hình kinh tế lượng
   - Biểu đồ trực quan hóa
   - Phân tích Machine Learning
   - Báo cáo tổng hợp và đề xuất

💡 Lưu ý: Script này được thiết kế để xử lý dữ liệu panel với cấu trúc
quốc gia-thời gian và các biến ICT, Intellectual Capital theo nghiên cứu của bạn.
""")
