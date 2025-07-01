# ICT & INTELLECTUAL CAPITAL COMPREHENSIVE ANALYSIS
# Phần 1-5: Từ Data Preprocessing đến Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import umap
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from kneed import KneeLocator

warnings.filterwarnings("ignore")

# Plotting setup
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# ====================================================================
# PHẦN 1: TIỀN XỬ LÝ VÀ TẠO CHỈ SỐ
# ====================================================================


class ICTAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        # Identify all numeric columns except for 'year' and 'country' columns
        exclude_cols = ["year", "country", "country_name"]
        self.all_numeric_vars = [
            col
            for col in self.df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        # Original variable groups
        self.ict_vars = ["inet_usr", "mob_sub", "ict_exp", "sec_srv"]
        self.ic_vars = ["edu_exp", "sci_art", "fdi", "trade"]
        # Use all numeric variables for analysis, not just ict/ic
        self.numeric_vars = self.all_numeric_vars
        self.scaler = StandardScaler()

    def descriptive_analysis(self):
        """1.1 Phân tích mô tả trước chuẩn hóa"""
        print("=" * 60)
        print("PHẦN 1.1: PHÂN TÍCH MÔ TẢ TRƯỚC CHUẨN HÓA")
        print("=" * 60)

        # Descriptive statistics
        desc_stats = self.df[self.numeric_vars].describe()
        print("\nDescriptive Statistics:")
        print(desc_stats)

        # Distribution plots
        n_vars = len(self.numeric_vars)
        n_cols = 4
        n_rows = int(np.ceil(n_vars / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle("Distribution of Variables Before Standardization", fontsize=16)

        # Flatten axes for easy indexing
        axes = axes.flatten() if n_vars > 1 else [axes]

        for i, var in enumerate(self.numeric_vars):
            ax = axes[i]
            ax.hist(
                self.df[var].dropna(),
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax.set_title(f"{var}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

            # Add normal curve
            mu, sigma = self.df[var].mean(), self.df[var].std()
            x = np.linspace(self.df[var].min(), self.df[var].max(), 100)
            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(
                x,
                y
                * len(self.df[var])
                * ((self.df[var].max() - self.df[var].min()) / 30),
                "r-",
                linewidth=2,
                label="Normal",
            )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Correlation heatmap before standardization
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[self.numeric_vars].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Correlation Matrix - Before Standardization")
        plt.show()

        return desc_stats, corr_matrix

    def standardize_data(self):
        """1.2 Chuẩn hóa z-score"""
        print("\n" + "=" * 60)
        print("PHẦN 1.2: CHUẨN HÓA DỮ LIỆU")
        print("=" * 60)

        # Standardization
        self.df_std = self.df.copy()
        self.df_std[self.numeric_vars] = self.scaler.fit_transform(
            self.df[self.numeric_vars]
        )

        # Add suffix for standardized variables
        std_vars = [f"{var}_z" for var in self.numeric_vars]
        self.df_std[std_vars] = self.df_std[self.numeric_vars]

        # Distribution plots after standardization
        n_vars = len(self.numeric_vars)
        n_cols = 4
        n_rows = int(np.ceil(n_vars / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle("Distribution of Variables After Standardization", fontsize=16)

        axes = axes.flatten() if n_vars > 1 else [axes]

        for i, var in enumerate(self.numeric_vars):
            axes[i].hist(
                self.df_std[var],
                bins=30,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
            )
            axes[i].set_title(f"{var} (Standardized)")
            axes[i].axvline(0, color="red", linestyle="--", alpha=0.7)
            axes[i].set_xlabel("Z-Score")
            axes[i].set_ylabel("Frequency")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Correlation heatmap after standardization
        plt.figure(figsize=(12, 10))
        corr_std = self.df_std[self.numeric_vars].corr()
        sns.heatmap(
            corr_std, annot=True, cmap="RdYlBu_r", center=0, square=True, linewidths=0.5
        )
        plt.title("Correlation Matrix - After Standardization")
        plt.show()

        # Scatter matrix
        from pandas.plotting import scatter_matrix

        scatter_matrix(self.df_std[self.numeric_vars], figsize=(15, 15), alpha=0.6)
        plt.suptitle("Scatter Matrix - Standardized Variables", fontsize=16)
        plt.show()

        return self.df_std

    def detect_outliers(self):
        """1.3 Phát hiện outliers"""
        print("\n" + "=" * 60)
        print("PHẦN 1.3: PHÁT HIỆN OUTLIERS")
        print("=" * 60)

        # Boxplots for outlier detection
        n_vars = len(self.numeric_vars)
        n_cols = 4
        n_rows = int(np.ceil(n_vars / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle("Outlier Detection - Boxplots", fontsize=16)

        axes = axes.flatten() if n_vars > 1 else [axes]

        outlier_counts = {}
        for i, var in enumerate(self.numeric_vars):
            axes[i].boxplot(self.df_std[var].dropna())
            axes[i].set_title(f"{var}")
            axes[i].set_ylabel("Z-Score")

            # Count outliers using IQR method
            Q1 = self.df_std[var].quantile(0.25)
            Q3 = self.df_std[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df_std[
                (self.df_std[var] < Q1 - 1.5 * IQR)
                | (self.df_std[var] > Q3 + 1.5 * IQR)
            ]
            outlier_counts[var] = len(outliers)

        plt.tight_layout()
        plt.show()

        print("Outlier counts by variable:")
        for var, count in outlier_counts.items():
            print(f"{var}: {count} outliers")

        return outlier_counts

    def create_ict_index_pca(self):
        """1.4 Tạo ICT Index bằng PCA"""
        print("\n" + "=" * 60)
        print("PHẦN 1.4: TẠO ICT INDEX BẰNG PCA")
        print("=" * 60)

        # KMO and Bartlett's test
        from factor_analyzer.factor_analyzer import (
            calculate_kmo,
            calculate_bartlett_sphericity,
        )

        ict_data = self.df_std[self.ict_vars].dropna()

        # KMO Test
        kmo_all, kmo_model = calculate_kmo(ict_data)
        print(f"KMO Test: {kmo_model:.4f}")

        # Bartlett's Test
        chi_square_value, p_value = calculate_bartlett_sphericity(ict_data)
        print(
            f"Bartlett's Test: Chi-square = {chi_square_value:.4f}, p-value = {p_value:.4f}"
        )

        if kmo_model > 0.6 and p_value < 0.05:
            print("✓ PCA is appropriate for this dataset")
        else:
            print("⚠ PCA may not be appropriate for this dataset")

        # PCA Analysis
        pca_ict = PCA()
        pca_ict.fit(ict_data)

        # Eigenvalues and explained variance
        eigenvalues = pca_ict.explained_variance_
        explained_var_ratio = pca_ict.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)

        print(f"\nEigenvalues: {eigenvalues}")
        print(f"Explained Variance Ratio: {explained_var_ratio}")
        print(f"Cumulative Variance Ratio: {cumulative_var_ratio}")

        # Scree plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-")
        plt.axhline(y=1, color="r", linestyle="--", alpha=0.7)
        plt.xlabel("Component")
        plt.ylabel("Eigenvalue")
        plt.title("Scree Plot - ICT Variables")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, "ro-")
        plt.axhline(y=0.8, color="g", linestyle="--", alpha=0.7, label="80% threshold")
        plt.xlabel("Component")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance - ICT")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Create ICT Index using PC1
        n_components = sum(eigenvalues > 1)
        print(f"\nNumber of components with eigenvalue > 1: {n_components}")

        pca_final = PCA(n_components=1)  # Use PC1 only
        ict_components = pca_final.fit_transform(ict_data)

        # ICT Index formula
        self.df_std["ICT_Index"] = np.nan
        self.df_std.loc[ict_data.index, "ICT_Index"] = ict_components.flatten()

        # PCA Loadings
        loadings = pca_final.components_.T
        loadings_df = pd.DataFrame(loadings, index=self.ict_vars, columns=["PC1"])

        print("PCA Loadings for ICT Index:")
        print(loadings_df)

        # Loadings plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(self.ict_vars)), loadings_df["PC1"])
        plt.yticks(range(len(self.ict_vars)), self.ict_vars)
        plt.xlabel("Loading")
        plt.title("PCA Loadings - ICT Index")
        plt.grid(True, alpha=0.3)
        plt.show()

        return loadings_df, pca_final

    def create_ic_index_hierarchical(self):
        """1.5 Tạo IC Index bằng phương pháp phân cấp"""
        print("\n" + "=" * 60)
        print("PHẦN 1.5: TẠO IC INDEX BẰNG PHƯƠNG PHÁP PHÂN CẤP")
        print("=" * 60)

        # Create sub-indices
        self.df_std["HC_Index"] = self.df_std["sci_art"]  # Human Capital
        self.df_std["SC_Index"] = self.df_std["edu_exp"]  # Structural Capital
        self.df_std["RC_Index"] = (
            self.df_std["fdi"] + self.df_std["trade"]
        ) / 2  # Relational Capital

        # Create composite IC Index
        self.df_std["IC_Index"] = (
            self.df_std["HC_Index"] + self.df_std["SC_Index"] + self.df_std["RC_Index"]
        ) / 3

        # Correlation between sub-indices
        sub_indices = ["HC_Index", "SC_Index", "RC_Index", "IC_Index"]
        corr_sub = self.df_std[sub_indices].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_sub, annot=True, cmap="RdYlBu_r", center=0, square=True)
        plt.title("Correlation Matrix - IC Sub-indices")
        plt.show()

        # Distribution of indices
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        indices = ["ICT_Index", "HC_Index", "SC_Index", "RC_Index"]

        for i, idx in enumerate(indices):
            row, col = i // 2, i % 2
            axes[row, col].hist(
                self.df_std[idx].dropna(), bins=30, alpha=0.7, edgecolor="black"
            )
            axes[row, col].set_title(f"{idx} Distribution")
            axes[row, col].set_xlabel("Value")
            axes[row, col].set_ylabel("Frequency")
            axes[row, col].axvline(
                self.df_std[idx].mean(), color="red", linestyle="--", alpha=0.7
            )

        plt.tight_layout()
        plt.show()

        print("IC Index Construction Summary:")
        print("HC_Index = sci_art_z")
        print("SC_Index = edu_exp_z")
        print("RC_Index = (fdi_in_z + trade_open_z) / 2")
        print("IC_Index = (HC_Index + SC_Index + RC_Index) / 3")

        return sub_indices

    # ====================================================================
    # PHẦN 2: PHÂN TÍCH TƯƠNG QUAN VÀ ĐA CỘNG TUYẾN
    # ====================================================================

    def correlation_analysis(self):
        """2.1 Phân tích tương quan đa dạng"""
        print("\n" + "=" * 60)
        print("PHẦN 2.1: PHÂN TÍCH TƯƠNG QUAN ĐA DẠNG")
        print("=" * 60)

        # Calculate different correlation types
        pearson_corr = self.df_std[self.numeric_vars].corr(method="pearson")
        spearman_corr = self.df_std[self.numeric_vars].corr(method="spearman")
        kendall_corr = self.df_std[self.numeric_vars].corr(method="kendall")

        # Plot three correlation heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        # Pearson
        sns.heatmap(
            pearson_corr,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            ax=axes[0],
            cbar_kws={"shrink": 0.8},
        )
        axes[0].set_title("Pearson Correlation")

        # Spearman
        sns.heatmap(
            spearman_corr,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            ax=axes[1],
            cbar_kws={"shrink": 0.8},
        )
        axes[1].set_title("Spearman Correlation")

        # Kendall
        sns.heatmap(
            kendall_corr,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            ax=axes[2],
            cbar_kws={"shrink": 0.8},
        )
        axes[2].set_title("Kendall Correlation")

        plt.tight_layout()
        plt.show()

        # Partial correlation analysis
        from scipy.stats import pearsonr

        def partial_correlation(df, x, y, control):
            """Calculate partial correlation between x and y controlling for control variables"""
            # Residuals from regression of x on control variables
            x_resid = (
                df[x]
                - df[control].values
                @ np.linalg.lstsq(df[control].values, df[x], rcond=None)[0]
            )
            # Residuals from regression of y on control variables
            y_resid = (
                df[y]
                - df[control].values
                @ np.linalg.lstsq(df[control].values, df[y], rcond=None)[0]
            )
            # Correlation between residuals
            return pearsonr(x_resid, y_resid)[0]

        # Example: Partial correlation between ICT and IC variables
        print("\nPartial Correlations (controlling for other variables):")
        for ict_var in self.ict_vars:
            for ic_var in self.ic_vars:
                control_vars = [
                    v for v in self.numeric_vars if v not in [ict_var, ic_var]
                ]
                try:
                    partial_corr = partial_correlation(
                        self.df_std[self.numeric_vars].dropna(),
                        ict_var,
                        ic_var,
                        control_vars,
                    )
                    print(f"{ict_var} - {ic_var}: {partial_corr:.4f}")
                except Exception:
                    print(f"{ict_var} - {ic_var}: Cannot compute")

        return pearson_corr, spearman_corr, kendall_corr

    def multicollinearity_analysis(self):
        """2.2 Phân tích đa cộng tuyến"""
        print("\n" + "=" * 60)
        print("PHẦN 2.2: PHÂN TÍCH ĐA CỘNG TUYẾN")
        print("=" * 60)

        # VIF Analysis
        def calculate_vif(df):
            vif_data = pd.DataFrame()
            vif_data["Variable"] = df.columns
            vif_data["VIF"] = [
                variance_inflation_factor(df.values, i) for i in range(df.shape[1])
            ]
            return vif_data

        # Calculate VIF for all numeric variables
        df_vif = self.df_std[self.numeric_vars].dropna()
        vif_results = calculate_vif(df_vif)

        print("Variance Inflation Factor (VIF) Analysis:")
        print(vif_results.sort_values("VIF", ascending=False))

        # VIF visualization
        plt.figure(figsize=(12, 8))
        plt.barh(vif_results["Variable"], vif_results["VIF"])
        plt.axvline(
            x=5, color="red", linestyle="--", alpha=0.7, label="VIF = 5 (Warning)"
        )
        plt.axvline(
            x=10,
            color="darkred",
            linestyle="--",
            alpha=0.7,
            label="VIF = 10 (Critical)",
        )
        plt.xlabel("VIF Value")
        plt.title("Variance Inflation Factor by Variable")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Condition Index Analysis
        corr_matrix = df_vif.corr()
        eigenvals = np.linalg.eigvals(corr_matrix)
        condition_indices = np.sqrt(max(eigenvals) / eigenvals)

        print("Condition Index Analysis:")
        print(f"Eigenvalues: {eigenvals}")
        print(f"Condition Indices: {condition_indices}")
        print(f"Maximum Condition Index: {max(condition_indices):.4f}")

        if max(condition_indices) > 30:
            print("⚠ Strong multicollinearity detected (CI > 30)")
        elif max(condition_indices) > 15:
            print("⚠ Moderate multicollinearity detected (CI > 15)")
        else:
            print("✓ No serious multicollinearity detected")

        # High correlation pairs
        print("High Correlation Pairs (|r| > 0.8):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )
                    print(
                        f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.4f}"
                    )

        if not high_corr_pairs:
            print("No variable pairs with |r| > 0.8")

        return vif_results, condition_indices, high_corr_pairs

    # ====================================================================
    # PHẦN 3: PHÂN TÍCH XU HƯỚNG THỜI GIAN
    # ====================================================================

    def time_series_analysis(self):
        """3.1 Phân tích xu hướng thời gian"""
        print("\n" + "=" * 60)
        print("PHẦN 3.1: PHÂN TÍCH XU HƯỚNG THỜI GIAN")
        print("=" * 60)

        # Assuming 'year' and 'country' columns exist
        if "year" not in self.df.columns:
            print("⚠ 'year' column not found. Creating sample time series...")
            self.df["year"] = np.random.choice(range(2010, 2021), len(self.df))

        if "country_name" not in self.df.columns:
            print("⚠ 'country_name' column not found. Creating sample countries...")
            countries = [
                "USA",
                "China",
                "Germany",
                "Japan",
                "UK",
                "France",
                "India",
                "Brazil",
            ]
            self.df["country"] = np.random.choice(countries, len(self.df))

        # Time series visualization
        key_variables = ["ICT_Index", "IC_Index"] + self.numeric_vars[:4]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for i, var in enumerate(key_variables):
            if var in self.df_std.columns:
                # Group by year and calculate mean
                ts_data = self.df_std.groupby("year")[var].mean()

                axes[i].plot(ts_data.index, ts_data.values, marker="o", linewidth=2)
                axes[i].set_title(f"{var} - Time Trend")
                axes[i].set_xlabel("Year")
                axes[i].set_ylabel("Mean Value")
                axes[i].grid(True, alpha=0.3)

                # Add trend line
                z = np.polyfit(ts_data.index, ts_data.values, 1)
                p = np.poly1d(z)
                axes[i].plot(ts_data.index, p(ts_data.index), "r--", alpha=0.8)

        plt.tight_layout()
        plt.show()

        # Country-wise trends for top variables
        top_countries = self.df["country_name"].value_counts().head(5).index

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        variables_to_plot = ["ICT_Index", "IC_Index", "inet_usr", "edu_exp"]

        for i, var in enumerate(variables_to_plot):
            if var in self.df_std.columns:
                row, col = i // 2, i % 2

                for country in top_countries:
                    country_data = self.df_std[self.df_std["country_name"] == country]
                    if len(country_data) > 1:
                        ts_country = country_data.groupby("year")[var].mean()
                        axes[row, col].plot(
                            ts_country.index,
                            ts_country.values,
                            marker="o",
                            label=country,
                            alpha=0.8,
                        )

                axes[row, col].set_title(f"{var} - Country Comparison")
                axes[row, col].set_xlabel("Year")
                axes[row, col].set_ylabel("Value")
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # YoY Growth Analysis
        print("\nYear-over-Year Growth Analysis:")
        for var in ["ICT_Index", "IC_Index"]:
            if var in self.df_std.columns:
                ts_data = self.df_std.groupby("year")[var].mean()
                yoy_growth = ts_data.pct_change() * 100
                print(f"\n{var} YoY Growth:")
                print(yoy_growth.dropna())

        return ts_data, yoy_growth

    def decompose_time_series(self):
        """3.2 Phân tích decomposition"""
        print("\n" + "=" * 60)
        print("PHẦN 3.2: TIME SERIES DECOMPOSITION")
        print("=" * 60)

        # Create more detailed time series for decomposition
        if len(self.df_std.groupby("year").size()) >= 4:  # Need at least 4 periods
            for var in ["ICT_Index", "IC_Index"]:
                if var in self.df_std.columns:
                    # Aggregate by year
                    ts_data = self.df_std.groupby("year")[var].mean()

                    if len(ts_data) >= 4:
                        # Perform decomposition
                        decomposition = seasonal_decompose(
                            ts_data, model="additive", period=2
                        )

                        # Plot decomposition
                        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

                        decomposition.observed.plot(
                            ax=axes[0], title=f"{var} - Original"
                        )
                        decomposition.trend.plot(ax=axes[1], title="Trend")
                        decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
                        decomposition.resid.plot(ax=axes[3], title="Residual")

                        plt.tight_layout()
                        plt.show()

                        # Stationarity test
                        adf_result = adfuller(ts_data.dropna())
                        print(f"\nADF Stationarity Test for {var}:")
                        print(f"ADF Statistic: {adf_result[0]:.4f}")
                        print(f"p-value: {adf_result[1]:.4f}")
                        print(f"Critical Values: {adf_result[4]}")

                        if adf_result[1] <= 0.05:
                            print("✓ Series is stationary")
                        else:
                            print("⚠ Series is non-stationary")
        else:
            print("⚠ Insufficient time periods for decomposition")

    # ====================================================================
    # PHẦN 4: DIMENSIONALITY REDUCTION COMPREHENSIVE
    # ====================================================================

    def comprehensive_pca(self):
        """4.1 PCA Analysis toàn diện"""
        print("\n" + "=" * 60)
        print("PHẦN 4.1: COMPREHENSIVE PCA ANALYSIS")
        print("=" * 60)

        # PCA on all 8 standardized variables
        pca_data = self.df_std[self.numeric_vars].dropna()

        # Full PCA
        pca_full = PCA()
        pca_full.fit(pca_data)

        # Explained variance analysis
        explained_var = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Scree plot and Kaiser criterion
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Scree plot
        axes[0].plot(
            range(1, len(explained_var) + 1), explained_var, "bo-", markersize=8
        )
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].set_title("Scree Plot")
        axes[0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[1].plot(
            range(1, len(cumulative_var) + 1), cumulative_var, "ro-", markersize=8
        )
        axes[1].axhline(
            y=0.8, color="g", linestyle="--", alpha=0.7, label="80% threshold"
        )
        axes[1].axhline(
            y=0.9, color="orange", linestyle="--", alpha=0.7, label="90% threshold"
        )
        axes[1].set_xlabel("Principal Component")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Explained Variance")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Eigenvalues
        eigenvalues = pca_full.explained_variance_
        axes[2].bar(range(1, len(eigenvalues) + 1), eigenvalues)
        axes[2].axhline(
            y=1, color="r", linestyle="--", alpha=0.7, label="Kaiser criterion (λ=1)"
        )
        axes[2].set_xlabel("Principal Component")
        axes[2].set_ylabel("Eigenvalue")
        axes[2].set_title("Eigenvalues")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Determine optimal number of components
        n_components_80 = np.argmax(cumulative_var >= 0.8) + 1
        n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
        n_components_kaiser = sum(eigenvalues > 1)

        print(f"Components needed for 80% variance: {n_components_80}")
        print(f"Components needed for 90% variance: {n_components_90}")
        print(f"Components with eigenvalue > 1 (Kaiser): {n_components_kaiser}")

        # Use optimal number of components
        n_optimal = (
            min(n_components_80, n_components_kaiser)
            if n_components_kaiser > 0
            else n_components_80
        )

        # Final PCA with optimal components
        pca_optimal = PCA(n_components=n_optimal)
        pca_scores = pca_optimal.fit_transform(pca_data)

        # PCA Biplot
        self.create_pca_biplot(pca_optimal, pca_scores, pca_data)

        # Loading analysis
        self.analyze_pca_loadings(pca_optimal)

        # PCA visualization with different color coding
        self.visualize_pca_scores(pca_scores, pca_data)

        return pca_optimal, pca_scores

    def create_pca_biplot(self, pca_model, scores, data):
        """Create PCA biplot"""
        plt.figure(figsize=(14, 10))

        # Plot scores
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.6, s=50)

        # Plot loadings as arrows
        loadings = pca_model.components_.T
        scale_factor = 3  # Scale arrows for visibility

        for i, (variable, loading) in enumerate(zip(self.numeric_vars, loadings)):
            plt.arrow(
                0,
                0,
                loading[0] * scale_factor,
                loading[1] * scale_factor,
                head_width=0.05,
                head_length=0.05,
                fc="red",
                ec="red",
                alpha=0.8,
            )
            plt.text(
                loading[0] * scale_factor * 1.1,
                loading[1] * scale_factor * 1.1,
                variable,
                fontsize=10,
                ha="center",
                va="center",
            )

        plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
        plt.title("PCA Biplot - Variables and Observations")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)
        plt.show()

    def analyze_pca_loadings(self, pca_model):
        """Analyze PCA loadings in detail"""
        loadings = pca_model.components_.T
        n_components = loadings.shape[1]

        # Create loadings dataframe
        loading_df = pd.DataFrame(
            loadings,
            index=self.numeric_vars,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )

        print("PCA Loadings Matrix:")
        print(loading_df.round(4))

        # Detailed loading plots
        fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 8))
        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            axes[i].barh(range(len(self.numeric_vars)), loading_df.iloc[:, i])
            axes[i].set_yticks(range(len(self.numeric_vars)))
            axes[i].set_yticklabels(self.numeric_vars)
            axes[i].set_xlabel("Loading")
            axes[i].set_title(
                f"PC{i + 1} Loadings\n({pca_model.explained_variance_ratio_[i]:.2%} variance)"
            )
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color="k", linestyle="-", alpha=0.5)

        plt.tight_layout()
        plt.show()

        # Contribution analysis
        print("Top contributors to each PC:")
        for i in range(n_components):
            pc_loadings = loading_df.iloc[:, i].abs().sort_values(ascending=False)
            print(f"\nPC{i + 1}:")
            for var, loading in pc_loadings.head(3).items():
                print(f"  {var}: {loading:.4f}")

        return loading_df

    def visualize_pca_scores(self, scores, data):
        """Visualize PCA scores with different color coding"""
        # Assuming we have additional variables for color coding
        if "ln_gdp" in self.df.columns:
            color_var = self.df.loc[data.index, "ln_gdp"]
            color_label = "ln_gdp"
        elif "hdi" in self.df.columns:
            color_var = self.df.loc[data.index, "hdi"]
            color_label = "hdi"
        else:
            color_var = scores[:, 0]  # Use PC1 as color
            color_label = "PC1"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # PC1 vs PC2 scatter
        scatter = axes[0].scatter(
            scores[:, 0], scores[:, 1], c=color_var, cmap="viridis", alpha=0.7, s=50
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title(f"PCA Scores colored by {color_label}")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0])

        # 3D plot if we have PC3
        if scores.shape[1] >= 3:
            ax = fig.add_subplot(122, projection="3d")
            scatter3d = ax.scatter(
                scores[:, 0],
                scores[:, 1],
                scores[:, 2],
                c=color_var,
                cmap="viridis",
                alpha=0.7,
                s=50,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"3D PCA Scores colored by {color_label}")
            plt.colorbar(scatter3d, ax=ax)
        else:
            # Time evolution if year data is available
            if "year" in self.df.columns:
                year_var = self.df.loc[data.index, "year"]
                scatter2 = axes[1].scatter(
                    scores[:, 0],
                    scores[:, 1],
                    c=year_var,
                    cmap="plasma",
                    alpha=0.7,
                    s=50,
                )
                axes[1].set_xlabel("PC1")
                axes[1].set_ylabel("PC2")
                axes[1].set_title("PCA Scores colored by Year")
                axes[1].grid(True, alpha=0.3)
                plt.colorbar(scatter2, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def alternative_dimensionality_reduction(self):
        """4.2 Alternative dimensionality reduction methods"""
        print("\n" + "=" * 60)
        print("PHẦN 4.2: ALTERNATIVE DIMENSIONALITY REDUCTION")
        print("=" * 60)

        data = self.df_std[self.numeric_vars].dropna()

        # t-SNE with different perplexity values
        perplexity_values = [5, 10, 30, 50]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, perp in enumerate(perplexity_values):
            if len(data) > perp:  # Ensure we have enough samples
                tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
                tsne_result = tsne.fit_transform(data)

                axes[i].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
                axes[i].set_title(f"t-SNE (perplexity={perp})")
                axes[i].set_xlabel("t-SNE 1")
                axes[i].set_ylabel("t-SNE 2")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # UMAP
        try:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_reducer.fit_transform(data)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.7)
            plt.title("UMAP Projection")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.grid(True, alpha=0.3)

            # Factor Analysis
            fa = FactorAnalysis(n_components=2, random_state=42)
            fa_result = fa.fit_transform(data)

            plt.subplot(1, 2, 2)
            plt.scatter(fa_result[:, 0], fa_result[:, 1], alpha=0.7)
            plt.title("Factor Analysis")
            plt.xlabel("Factor 1")
            plt.ylabel("Factor 2")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in UMAP/Factor Analysis: {e}")

        # Independent Component Analysis (ICA)
        try:
            ica = FastICA(n_components=2, random_state=42)
            ica_result = ica.fit_transform(data)

            # Multidimensional Scaling (MDS)
            mds = MDS(n_components=2, random_state=42)
            mds_result = mds.fit_transform(data)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].scatter(ica_result[:, 0], ica_result[:, 1], alpha=0.7)
            axes[0].set_title("Independent Component Analysis")
            axes[0].set_xlabel("IC 1")
            axes[0].set_ylabel("IC 2")
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(mds_result[:, 0], mds_result[:, 1], alpha=0.7)
            axes[1].set_title("Multidimensional Scaling")
            axes[1].set_xlabel("MDS 1")
            axes[1].set_ylabel("MDS 2")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in ICA/MDS: {e}")

    # ====================================================================
    # PHẦN 5: CLUSTERING VÀ PATTERN RECOGNITION
    # ====================================================================

    def advanced_clustering(self):
        """5.1 Advanced clustering methods"""
        print("\n" + "=" * 60)
        print("PHẦN 5.1: ADVANCED CLUSTERING METHODS")
        print("=" * 60)

        data = self.df_std[self.numeric_vars].dropna()

        # Optimal K selection for K-means
        self.find_optimal_k(data)

        # K-means clustering with optimal k
        optimal_k = self.determine_optimal_k(data)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)

        # Hierarchical clustering
        hierarchical_labels = self.hierarchical_clustering(data)

        # DBSCAN
        dbscan_labels = self.dbscan_clustering(data)

        # Gaussian Mixture Model
        gmm_labels = self.gmm_clustering(data)

        # Spectral clustering
        spectral_labels = self.spectral_clustering(data, optimal_k)

        # Store all clustering results
        self.clustering_results = {
            "kmeans": kmeans_labels,
            "hierarchical": hierarchical_labels,
            "dbscan": dbscan_labels,
            "gmm": gmm_labels,
            "spectral": spectral_labels,
        }

        # Visualize clustering results
        self.visualize_clustering_results(data)

        return self.clustering_results

    def find_optimal_k(self, data):
        """Find optimal number of clusters using multiple methods"""
        max_k = min(10, len(data) // 2)
        k_range = range(2, max_k + 1)

        # Initialize metrics
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(data)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(data, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(data, cluster_labels))

        # Plot optimization metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Elbow method
        axes[0, 0].plot(k_range, inertias, "bo-")
        axes[0, 0].set_xlabel("Number of Clusters (k)")
        axes[0, 0].set_ylabel("Inertia")
        axes[0, 0].set_title("Elbow Method")
        axes[0, 0].grid(True, alpha=0.3)

        # Find elbow using KneeLocator
        try:
            kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            elbow_k = kl.elbow
            if elbow_k:
                axes[0, 0].axvline(
                    x=elbow_k, color="r", linestyle="--", label=f"Elbow at k={elbow_k}"
                )
                axes[0, 0].legend()
        except Exception:
            elbow_k = None

        # Silhouette analysis
        axes[0, 1].plot(k_range, silhouette_scores, "go-")
        axes[0, 1].set_xlabel("Number of Clusters (k)")
        axes[0, 1].set_ylabel("Silhouette Score")
        axes[0, 1].set_title("Silhouette Analysis")
        axes[0, 1].grid(True, alpha=0.3)

        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        axes[0, 1].axvline(
            x=best_silhouette_k,
            color="r",
            linestyle="--",
            label=f"Best k={best_silhouette_k}",
        )
        axes[0, 1].legend()

        # Calinski-Harabasz Index
        axes[1, 0].plot(k_range, calinski_scores, "mo-")
        axes[1, 0].set_xlabel("Number of Clusters (k)")
        axes[1, 0].set_ylabel("Calinski-Harabasz Score")
        axes[1, 0].set_title("Calinski-Harabasz Index")
        axes[1, 0].grid(True, alpha=0.3)

        best_calinski_k = k_range[np.argmax(calinski_scores)]
        axes[1, 0].axvline(
            x=best_calinski_k,
            color="r",
            linestyle="--",
            label=f"Best k={best_calinski_k}",
        )
        axes[1, 0].legend()

        # Davies-Bouldin Index
        axes[1, 1].plot(k_range, davies_bouldin_scores, "co-")
        axes[1, 1].set_xlabel("Number of Clusters (k)")
        axes[1, 1].set_ylabel("Davies-Bouldin Score")
        axes[1, 1].set_title("Davies-Bouldin Index")
        axes[1, 1].grid(True, alpha=0.3)

        best_davies_k = k_range[np.argmin(davies_bouldin_scores)]
        axes[1, 1].axvline(
            x=best_davies_k, color="r", linestyle="--", label=f"Best k={best_davies_k}"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        # Summary of optimal k suggestions
        print("Optimal K Suggestions:")
        if elbow_k:
            print(f"Elbow Method: k = {elbow_k}")
        print(f"Silhouette Score: k = {best_silhouette_k}")
        print(f"Calinski-Harabasz: k = {best_calinski_k}")
        print(f"Davies-Bouldin: k = {best_davies_k}")

        return elbow_k, best_silhouette_k, best_calinski_k, best_davies_k

    def determine_optimal_k(self, data):
        """Determine single optimal k value"""
        suggestions = self.find_optimal_k(data)
        # Use mode of suggestions, or default to 3
        valid_suggestions = [k for k in suggestions if k is not None]
        if valid_suggestions:
            return max(set(valid_suggestions), key=valid_suggestions.count)
        else:
            return 3

    def hierarchical_clustering(self, data):
        """Perform hierarchical clustering"""
        print("\nHierarchical Clustering:")

        # Different linkage methods
        linkage_methods = ["ward", "complete", "average"]

        fig, axes = plt.subplots(1, len(linkage_methods), figsize=(20, 6))

        for i, method in enumerate(linkage_methods):
            # Perform linkage
            Z = linkage(data, method=method)

            # Plot dendrogram
            dendrogram(Z, ax=axes[i], truncate_mode="level", p=5)
            axes[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")
            axes[i].set_xlabel("Sample Index")
            axes[i].set_ylabel("Distance")

        plt.tight_layout()
        plt.show()

        # Use ward linkage for final clustering
        Z_ward = linkage(data, method="ward")
        hierarchical_labels = fcluster(Z_ward, t=3, criterion="maxclust")

        print(
            f"Hierarchical clustering completed with {len(np.unique(hierarchical_labels))} clusters"
        )

        return hierarchical_labels

    def dbscan_clustering(self, data):
        """Perform DBSCAN clustering with parameter tuning"""
        print("\nDBSCAN Clustering:")

        # Parameter tuning for DBSCAN
        eps_values = [0.3, 0.5, 0.7, 1.0]
        min_samples_values = [3, 5, 7]

        best_score = -1
        best_params = None
        best_labels = None

        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)

                if len(np.unique(labels)) > 1:  # More than just noise
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        best_labels = labels

        if best_params:
            print(
                f"Best DBSCAN parameters: eps={best_params[0]}, min_samples={best_params[1]}"
            )
            print(f"Best silhouette score: {best_score:.4f}")
            print(
                f"Number of clusters: {len(np.unique(best_labels[best_labels != -1]))}"
            )
            print(f"Number of noise points: {sum(best_labels == -1)}")
        else:
            print("DBSCAN could not find meaningful clusters with tested parameters")
            best_labels = np.zeros(len(data))  # Fallback

        return best_labels

    def gmm_clustering(self, data):
        """Perform Gaussian Mixture Model clustering"""
        print("\nGaussian Mixture Model Clustering:")

        # Test different numbers of components
        n_components_range = range(2, 8)
        aic_scores = []
        bic_scores = []

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(data)
            aic_scores.append(gmm.aic(data))
            bic_scores.append(gmm.bic(data))

        # Plot AIC and BIC
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(n_components_range, aic_scores, "bo-", label="AIC")
        plt.xlabel("Number of Components")
        plt.ylabel("AIC Score")
        plt.title("AIC vs Number of Components")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(n_components_range, bic_scores, "ro-", label="BIC")
        plt.xlabel("Number of Components")
        plt.ylabel("BIC Score")
        plt.title("BIC vs Number of Components")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Choose optimal number of components (lowest BIC)
        optimal_n = n_components_range[np.argmin(bic_scores)]
        print(f"Optimal number of components (BIC): {optimal_n}")

        # Final GMM
        gmm_final = GaussianMixture(n_components=optimal_n, random_state=42)
        gmm_labels = gmm_final.fit_predict(data)

        return gmm_labels

    def spectral_clustering(self, data, n_clusters):
        """Perform spectral clustering"""
        print(f"\nSpectral Clustering with {n_clusters} clusters:")

        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
        spectral_labels = spectral.fit_predict(data)

        return spectral_labels

    def visualize_clustering_results(self, data):
        """Visualize all clustering results"""
        print("\n" + "=" * 60)
        print("CLUSTERING RESULTS VISUALIZATION")
        print("=" * 60)

        # Use PCA for 2D visualization
        pca_viz = PCA(n_components=2)
        data_2d = pca_viz.fit_transform(data)

        # Plot all clustering results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        methods = ["kmeans", "hierarchical", "dbscan", "gmm", "spectral"]

        for i, method in enumerate(methods):
            if method in self.clustering_results:
                labels = self.clustering_results[method]

                # Create scatter plot
                axes[i].scatter(
                    data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
                )
                axes[i].set_title(f"{method.upper()} Clustering")
                axes[i].set_xlabel("PC1")
                axes[i].set_ylabel("PC2")
                axes[i].grid(True, alpha=0.3)

                # Add cluster centers for K-means
                if method == "kmeans":
                    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
                    kmeans.fit(data)
                    centers_2d = pca_viz.transform(kmeans.cluster_centers_)
                    axes[i].scatter(
                        centers_2d[:, 0],
                        centers_2d[:, 1],
                        c="red",
                        marker="x",
                        s=200,
                        linewidths=3,
                    )

        # Remove empty subplot
        if len(methods) < len(axes):
            axes[-1].remove()

        plt.tight_layout()
        plt.show()

    def cluster_validation_analysis(self):
        """5.3 Cluster validation and interpretation"""
        print("\n" + "=" * 60)
        print("PHẦN 5.3: CLUSTER VALIDATION & INTERPRETATION")
        print("=" * 60)

        data = self.df_std[self.numeric_vars].dropna()

        # Calculate validation metrics for each clustering method
        validation_results = {}

        for method, labels in self.clustering_results.items():
            if len(np.unique(labels)) > 1:  # Skip if only one cluster
                try:
                    silhouette = silhouette_score(data, labels)
                    davies_bouldin = davies_bouldin_score(data, labels)
                    calinski_harabasz = calinski_harabasz_score(data, labels)

                    validation_results[method] = {
                        "silhouette": silhouette,
                        "davies_bouldin": davies_bouldin,
                        "calinski_harabasz": calinski_harabasz,
                        "n_clusters": len(np.unique(labels)),
                    }
                except Exception:
                    validation_results[method] = {
                        "silhouette": 0,
                        "davies_bouldin": float("inf"),
                        "calinski_harabasz": 0,
                        "n_clusters": len(np.unique(labels)),
                    }

        # Create validation summary
        validation_df = pd.DataFrame(validation_results).T
        print("Clustering Validation Summary:")
        print(validation_df.round(4))

        # Visualize validation metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        methods = list(validation_results.keys())
        silhouette_scores = [validation_results[m]["silhouette"] for m in methods]
        davies_bouldin_scores = [
            validation_results[m]["davies_bouldin"] for m in methods
        ]
        calinski_scores = [validation_results[m]["calinski_harabasz"] for m in methods]

        axes[0].bar(methods, silhouette_scores, color="skyblue")
        axes[0].set_title("Silhouette Score (Higher is Better)")
        axes[0].set_ylabel("Score")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(methods, davies_bouldin_scores, color="lightcoral")
        axes[1].set_title("Davies-Bouldin Score (Lower is Better)")
        axes[1].set_ylabel("Score")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(methods, calinski_scores, color="lightgreen")
        axes[2].set_title("Calinski-Harabasz Score (Higher is Better)")
        axes[2].set_ylabel("Score")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        # Cluster profiling
        self.cluster_profiling(data)

        return validation_df

    def cluster_profiling(self, data):
        """Profile clusters by calculating mean of each variable per cluster for each method."""
        print("\nCluster Profiling:")
        for method, labels in self.clustering_results.items():
            if len(np.unique(labels)) > 1:
                print(f"\nMethod: {method.upper()}")
                df_profile = data.copy()
                df_profile["cluster"] = labels
                profile_means = df_profile.groupby("cluster").mean()
                print(profile_means)


def run_and_save_processed_dataset(input_csv, output_csv="EDA_2.csv"):
    # Load data
    df = pd.read_csv(input_csv)
    analyzer = ICTAnalyzer(df)
    # Run all main steps
    analyzer.descriptive_analysis()
    analyzer.standardize_data()
    analyzer.detect_outliers()
    analyzer.create_ict_index_pca()
    analyzer.create_ic_index_hierarchical()
    analyzer.correlation_analysis()
    analyzer.multicollinearity_analysis()
    analyzer.time_series_analysis()
    analyzer.decompose_time_series()
    analyzer.comprehensive_pca()
    analyzer.alternative_dimensionality_reduction()
    analyzer.advanced_clustering()
    analyzer.cluster_validation_analysis()

    # Save processed DataFrame (Optional)
    # analyzer.df_std.to_csv(output_csv, index=False)

    print("Đã hoàn thành phân tích và lưu trữ dữ liệu đã xử lý.")


# Example usage:
run_and_save_processed_dataset(
    r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\processed_dataset.csv"
)
