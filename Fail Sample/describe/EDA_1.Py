# COMPREHENSIVE ICT & INTELLECTUAL CAPITAL DATA ANALYSIS
# =======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import anderson, kstest
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
)
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import missingno as msno
import umap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ICTAnalyzer:
    """
    Comprehensive analyzer for ICT & Intellectual Capital dataset
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize analyzer with data
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")

        # Convert all columns except 'country_name', 'country_code', 'year' to numeric (force errors to NaN)
        cols_to_numeric = [
            col
            for col in self.df.columns
            if col not in ["country_name", "country_code", "year"]
        ]
        self.df[cols_to_numeric] = self.df[cols_to_numeric].apply(
            pd.to_numeric, errors="coerce"
        )
        self.numeric_cols = [
            col
            for col in self.df.select_dtypes(include=[np.number]).columns.tolist()
            if col not in ["year"]
        ]
        # Define target variables for ICT & Intellectual Capital
        self.ict_vars = ["inet_usr", "mob_sub", "ict_exp", "sec_srv"]
        self.ic_vars = ["edu_exp", "sci_art", "fdi", "trade"]
        self.target_vars = self.ict_vars + self.ic_vars
        self.scalers = {}
        self.scaled_data = {}

        print("ICT & Intellectual Capital Analyzer initialized")
        print(f"Dataset shape: {self.df.shape}")

    def basic_info(self):
        """
        PHẦN 1.1: Thống kê mô tả chi tiết
        """
        print("=" * 60)
        print("PHẦN 1.1: THỐNG KÊ MÔ TẢ CHI TIẾT")
        print("=" * 60)

        # Basic info
        print("\n1.1.1 Dataset Info:")
        print(self.df.info())

        # Identify numeric columns
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns: {self.numeric_cols}")

        # Extended descriptive statistics
        print("\n1.1.2 Extended Descriptive Statistics:")
        percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        desc_stats = self.df[self.numeric_cols].describe(percentiles=percentiles)
        print(desc_stats)

        # Skewness and Kurtosis
        print("\n1.1.3 Skewness and Kurtosis Analysis:")
        skew_kurt_df = pd.DataFrame(
            {
                "Variable": self.numeric_cols,
                "Skewness": [self.df[col].skew() for col in self.numeric_cols],
                "Kurtosis": [self.df[col].kurtosis() for col in self.numeric_cols],
                "Skew_Interpretation": [""] * len(self.numeric_cols),
                "Kurt_Interpretation": [""] * len(self.numeric_cols),
            }
        )

        # Interpret skewness and kurtosis
        for i, (skew, kurt) in enumerate(
            zip(skew_kurt_df["Skewness"], skew_kurt_df["Kurtosis"])
        ):
            # Skewness interpretation
            if abs(skew) < 0.5:
                skew_kurt_df.loc[i, "Skew_Interpretation"] = "Symmetric"
            elif 0.5 <= abs(skew) < 1:
                skew_kurt_df.loc[i, "Skew_Interpretation"] = "Moderate Skew"
            else:
                skew_kurt_df.loc[i, "Skew_Interpretation"] = "High Skew"

            # Kurtosis interpretation
            if kurt < 3:
                skew_kurt_df.loc[i, "Kurt_Interpretation"] = "Platykurtic"
            elif kurt > 3:
                skew_kurt_df.loc[i, "Kurt_Interpretation"] = "Leptokurtic"
            else:
                skew_kurt_df.loc[i, "Kurt_Interpretation"] = "Mesokurtic"

        print(skew_kurt_df.round(3))

        # Outlier detection
        print("\n1.1.4 Outlier Detection:")
        outlier_summary = self.detect_outliers()

        return desc_stats, skew_kurt_df, outlier_summary

    def detect_outliers(self):
        """
        Detect outliers using IQR and Z-score methods
        """
        outlier_summary = []

        for col in self.numeric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()

                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = len(data[(data < lower_bound) | (data > upper_bound)])

                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_outliers = len(data[z_scores > 3])

                outlier_summary.append(
                    {
                        "Variable": col,
                        "Total_Records": len(data),
                        "IQR_Outliers": iqr_outliers,
                        "IQR_Percentage": (iqr_outliers / len(data)) * 100,
                        "ZScore_Outliers": z_outliers,
                        "ZScore_Percentage": (z_outliers / len(data)) * 100,
                    }
                )

        outlier_df = pd.DataFrame(outlier_summary)
        print(outlier_df.round(2))
        return outlier_df

    def missing_data_analysis(self):
        """
        PHẦN 1.2: Phân tích dữ liệu thiếu nâng cao
        """
        print("\n" + "=" * 60)
        print("PHẦN 1.2: PHÂN TÍCH DỮ LIỆU THIẾU NÂNG CAO")
        print("=" * 60)

        # Missing data statistics
        missing_stats = pd.DataFrame(
            {
                "Variable": self.df.columns,
                "Missing_Count": self.df.isnull().sum(),
                "Missing_Percentage": (self.df.isnull().sum() / len(self.df)) * 100,
                "Data_Type": self.df.dtypes,
            }
        )
        missing_stats = missing_stats[missing_stats["Missing_Count"] > 0].sort_values(
            "Missing_Percentage", ascending=False
        )

        print("\n1.2.1 Missing Data Summary:")
        print(missing_stats)

        # Missing data visualizations
        if missing_stats.shape[0] > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Missing data matrix
            msno.matrix(self.df, ax=axes[0, 0])
            axes[0, 0].set_title("Missing Data Matrix")

            # Missing data heatmap
            msno.heatmap(self.df, ax=axes[0, 1])
            axes[0, 1].set_title("Missing Data Correlation Heatmap")

            # Missing data bar plot
            msno.bar(self.df, ax=axes[1, 0])
            axes[1, 0].set_title("Missing Data Count by Variable")

            # Missing data dendrogram
            msno.dendrogram(self.df, ax=axes[1, 1])
            axes[1, 1].set_title("Missing Data Dendrogram")

            plt.tight_layout()
            plt.show()

        # Missing data by country and year
        if "country_name" in self.df.columns and "year" in self.df.columns:
            print("\n1.2.2 Missing Data by Country (Top 10):")
            country_missing = (
                self.df.groupby("country_name")
                .apply(lambda x: x.isnull().sum().sum())
                .sort_values(ascending=False)
                .head(10)
            )
            print(country_missing)

            print("\n1.2.3 Missing Data by Year:")
            year_missing = (
                self.df.groupby("year")
                .apply(lambda x: x.isnull().sum().sum())
                .sort_values(ascending=False)
            )
            print(year_missing)

        return missing_stats

    def time_series_analysis(self):
        """
        PHẦN 1.3: Phân tích xu hướng thời gian
        """
        if "year" not in self.df.columns:
            print("No 'year' column found. Skipping time series analysis.")
            return

        print("\n" + "=" * 60)
        print("PHẦN 1.3: PHÂN TÍCH XU HƯỚNG THỜI GIAN")
        print("=" * 60)

        # Time series plots for key variables
        key_vars = [col for col in self.target_vars if col in self.df.columns]

        if len(key_vars) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()

            for i, var in enumerate(key_vars[:4]):
                # Aggregate by year (mean)
                yearly_data = self.df.groupby("year")[var].mean()
                axes[i].plot(yearly_data.index, yearly_data.values, marker="o")
                axes[i].set_title(f"{var} - Time Trend")
                axes[i].set_xlabel("Year")
                axes[i].set_ylabel(var)
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        # Growth rate analysis
        print("\n1.3.1 Growth Rate Analysis:")
        if "country_name" in self.df.columns:
            growth_rates = {}
            for var in key_vars:
                if var in self.df.columns:
                    country_growth = []
                    for country in self.df["country_name"].unique():
                        country_data = self.df[
                            self.df["country_name"] == country
                        ].sort_values("year")
                        if len(country_data) > 1:
                            growth = country_data[var].pct_change().mean() * 100
                            if not np.isnan(growth):
                                country_growth.append(growth)

                    if country_growth:
                        growth_rates[var] = {
                            "Mean_Growth": np.mean(country_growth),
                            "Median_Growth": np.median(country_growth),
                            "Std_Growth": np.std(country_growth),
                        }

            growth_df = pd.DataFrame(growth_rates).T
            print(growth_df.round(3))

    def correlation_analysis(self):
        """
        PHẦN 2: PHÂN TÍCH TƯƠNG QUAN VÀ RELATIONSHIPS
        """
        print("\n" + "=" * 60)
        print("PHẦN 2: PHÂN TÍCH TƯƠNG QUAN VÀ RELATIONSHIPS")
        print("=" * 60)

        numeric_data = self.df[self.numeric_cols]

        # 2.1 Multiple correlation matrices
        correlations = {}
        correlations["pearson"] = numeric_data.corr(method="pearson")
        correlations["spearman"] = numeric_data.corr(method="spearman")
        correlations["kendall"] = numeric_data.corr(method="kendall")

        # Plot correlation heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for i, (method, corr_matrix) in enumerate(correlations.items()):
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                ax=axes[i],
                fmt=".2f",
            )
            axes[i].set_title(f"{method.capitalize()} Correlation Matrix")

        plt.tight_layout()
        plt.show()

        # 2.2 Multicollinearity analysis
        print("\n2.2 Multicollinearity Analysis:")

        # VIF calculation
        vif_data = numeric_data.dropna()
        if len(vif_data) > 0:
            vif_df = pd.DataFrame()
            vif_df["Variable"] = vif_data.columns
            vif_df["VIF"] = [
                variance_inflation_factor(vif_data.values, i)
                for i in range(len(vif_data.columns))
            ]
            vif_df = vif_df.sort_values("VIF", ascending=False)
            print("Variance Inflation Factor (VIF):")
            print(vif_df)

            # High correlation pairs
            print("\n2.2.2 High Correlation Pairs (|r| > 0.8):")
            pearson_corr = correlations["pearson"]
            high_corr_pairs = []

            for i in range(len(pearson_corr.columns)):
                for j in range(i + 1, len(pearson_corr.columns)):
                    corr_val = pearson_corr.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append(
                            {
                                "Variable_1": pearson_corr.columns[i],
                                "Variable_2": pearson_corr.columns[j],
                                "Correlation": corr_val,
                            }
                        )

            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                print(high_corr_df.round(3))
            else:
                print("No high correlation pairs found.")

        return correlations, vif_df if "vif_df" in locals() else None

    def feature_engineering(self):
        """
        PHẦN 3: FEATURE ENGINEERING & TRANSFORMATION
        """
        print("\n" + "=" * 60)
        print("PHẦN 3: FEATURE ENGINEERING & TRANSFORMATION")
        print("=" * 60)

        # 3.1 Create new features
        print("\n3.1 Creating New Features:")

        # Log transformations for skewed variables
        skewed_vars = (
            ["gdp", "pop", "ict_exp", "edu_exp"]
            if all(
                col in self.df.columns for col in ["gdp", "pop", "ict_exp", "edu_exp"]
            )
            else []
        )
        for var in skewed_vars:
            if var in self.df.columns and (self.df[var] > 0).all():
                self.df[f"ln_{var}"] = np.log(self.df[var])
                print(f"Created ln_{var}")

        # Ratio variables
        if "gdp" in self.df.columns:
            ratio_vars = ["ict_exp", "edu_exp", "fdi", "trade"]
            for var in ratio_vars:
                if var in self.df.columns:
                    self.df[f"{var}_per_gdp"] = self.df[var] / self.df["gdp"]
                    print(f"Created {var}_per_gdp")

        # Interaction terms
        interactions = [("ict_exp", "edu_exp"), ("inet_usr", "sci_art")]
        for var1, var2 in interactions:
            if var1 in self.df.columns and var2 in self.df.columns:
                self.df[f"{var1}_x_{var2}"] = self.df[var1] * self.df[var2]
                print(f"Created {var1}_x_{var2}")

        # Growth rates (if time series data available)
        if "year" in self.df.columns and "country_name" in self.df.columns:
            growth_vars = ["gdp", "ict_exp", "edu_exp"]
            for var in growth_vars:
                if var in self.df.columns:
                    self.df = self.df.sort_values(["country_name", "year"])
                    self.df[f"{var}_growth"] = self.df.groupby("country_name")[
                        var
                    ].pct_change()
                    print(f"Created {var}_growth")

        # Update numeric columns
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nTotal features after engineering: {len(self.numeric_cols)}")

    def feature_selection(self, target_var=None):
        """
        PHẦN 3.2: Feature Selection Methods
        """
        if target_var is None or target_var not in self.df.columns:
            print("Skipping feature selection - no valid target variable specified")
            return

        print(f"\n3.2 Feature Selection for target: {target_var}")

        # Prepare data
        feature_cols = [col for col in self.numeric_cols if col != target_var]
        X = self.df[feature_cols].dropna()
        y = self.df.loc[X.index, target_var]

        if len(X) == 0:
            print("No data available for feature selection")
            return

        results = {}

        # Univariate statistical tests
        f_scores, f_pvalues = f_regression(X, y)
        results["f_regression"] = pd.DataFrame(
            {"Feature": feature_cols, "F_Score": f_scores, "P_Value": f_pvalues}
        ).sort_values("F_Score", ascending=False)

        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        results["mutual_info"] = pd.DataFrame(
            {"Feature": feature_cols, "MI_Score": mi_scores}
        ).sort_values("MI_Score", ascending=False)

        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        results["random_forest"] = pd.DataFrame(
            {"Feature": feature_cols, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=False)

        # Display top features
        print("\nTop 10 Features by F-Regression:")
        print(results["f_regression"].head(10))

        print("\nTop 10 Features by Mutual Information:")
        print(results["mutual_info"].head(10))

        print("\nTop 10 Features by Random Forest:")
        print(results["random_forest"].head(10))

        return results

    def scaling_analysis(self):
        """
        PHẦN 4: CHUẨN HÓA VÀ DISTRIBUTION ANALYSIS
        """
        print("\n" + "=" * 60)
        print("PHẦN 4: CHUẨN HÓA VÀ DISTRIBUTION ANALYSIS")
        print("=" * 60)

        # Select 8 main ICT & IC variables
        available_vars = [var for var in self.target_vars if var in self.df.columns]
        if len(available_vars) < len(self.target_vars):
            print(
                f"Warning: Only {len(available_vars)} out of {len(self.target_vars)} target variables found"
            )
            print(f"Available variables: {available_vars}")

        # Prepare data for scaling
        scaling_data = self.df[available_vars].dropna()

        if len(scaling_data) == 0:
            print("No complete data available for scaling analysis")
            return

        # Apply different scaling methods
        scaling_methods = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "QuantileTransformer_Normal": QuantileTransformer(
                output_distribution="normal"
            ),
            "QuantileTransformer_Uniform": QuantileTransformer(
                output_distribution="uniform"
            ),
        }

        print(f"\n4.1 Applying Scaling Methods to {len(available_vars)} variables:")
        print(f"Data shape for scaling: {scaling_data.shape}")

        for method_name, scaler in scaling_methods.items():
            try:
                scaled_data = scaler.fit_transform(scaling_data)
                self.scaled_data[method_name] = pd.DataFrame(
                    scaled_data, columns=available_vars, index=scaling_data.index
                )
                self.scalers[method_name] = scaler
                print(f"✓ {method_name} applied successfully")
            except Exception as e:
                print(f"✗ Error applying {method_name}: {str(e)}")

        # Distribution analysis after scaling (using StandardScaler as default)
        if "StandardScaler" in self.scaled_data:
            self.distribution_analysis(
                self.scaled_data["StandardScaler"], available_vars
            )

        return self.scaled_data

    def distribution_analysis(self, scaled_data, variables):
        """
        4.3 Distribution Analysis sau chuẩn hóa
        """
        print("\n4.3 Distribution Analysis after Scaling:")

        n_vars = len(variables)
        n_cols = min(4, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        # Histograms and density plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.ravel()

        for i, var in enumerate(variables):
            if i < len(axes):
                data = scaled_data[var].dropna()
                axes[i].hist(data, bins=30, density=True, alpha=0.7, color="skyblue")
                axes[i].axvline(
                    data.mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {data.mean():.2f}",
                )
                axes[i].axvline(
                    data.median(),
                    color="green",
                    linestyle="--",
                    label=f"Median: {data.median():.2f}",
                )
                axes[i].set_title(f"{var} - Scaled Distribution")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Normality tests
        print("\n4.3.1 Normality Tests Results:")
        normality_results = []

        for var in variables:
            data = scaled_data[var].dropna()
            if len(data) > 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data)

                # Anderson-Darling test
                anderson_result = anderson(data, dist="norm")

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, "norm")

                normality_results.append(
                    {
                        "Variable": var,
                        "Shapiro_Stat": shapiro_stat,
                        "Shapiro_P": shapiro_p,
                        "Shapiro_Normal": "Yes" if shapiro_p > 0.05 else "No",
                        "Anderson_Stat": anderson_result.statistic,
                        "Anderson_Critical_5%": anderson_result.critical_values[2],
                        "Anderson_Normal": "Yes"
                        if anderson_result.statistic
                        < anderson_result.critical_values[2]
                        else "No",
                        "KS_Stat": ks_stat,
                        "KS_P": ks_p,
                        "KS_Normal": "Yes" if ks_p > 0.05 else "No",
                    }
                )

        normality_df = pd.DataFrame(normality_results)
        print(normality_df.round(4))

        return normality_df

    def pca_analysis(
        self,
        scaling_method="StandardScaler",
        label_col="country_name",
        color_cols=["ln_gdp", "hdi", "year"],
    ):
        """
        PHẦN 5: DIMENSIONALITY REDUCTION COMPREHENSIVE
        """
        print("\n" + "=" * 60)
        print("PHẦN 5: DIMENSIONALITY REDUCTION COMPREHENSIVE")
        print("=" * 60)

        if scaling_method not in self.scaled_data:
            print(
                f"Scaling method {scaling_method} not available. Using first available method."
            )
            scaling_method = list(self.scaled_data.keys())[0]

        data = self.scaled_data[scaling_method]
        print(f"Using {scaling_method} for PCA analysis")
        print(f"Data shape: {data.shape}")

        # 5.1 Principal Component Analysis
        print("\n5.1 Principal Component Analysis:")

        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(data)

        # Explained variance analysis
        explained_var_ratio = pca_full.explained_variance_ratio_
        cumsum_var_ratio = np.cumsum(explained_var_ratio)

        print("Explained Variance by Component:")
        for i, (var, cumvar) in enumerate(zip(explained_var_ratio, cumsum_var_ratio)):
            print(f"PC{i + 1}: {var:.3f} ({cumvar:.3f} cumulative)")

        # Scree plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, "bo-")
        ax1.axhline(
            y=1 / len(data.columns),
            color="r",
            linestyle="--",
            label=f"Average eigenvalue (1/{len(data.columns)})",
        )
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Scree Plot")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(1, len(cumsum_var_ratio) + 1), cumsum_var_ratio, "ro-")
        ax2.axhline(y=0.8, color="g", linestyle="--", label="80% threshold")
        ax2.axhline(y=0.9, color="b", linestyle="--", label="90% threshold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance Ratio")
        ax2.set_title("Cumulative Explained Variance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Select number of components (80% variance or Kaiser criterion)
        n_components_80 = np.argmax(cumsum_var_ratio >= 0.8) + 1
        n_components_kaiser = np.sum(pca_full.explained_variance_ > 1)
        n_components = max(2, min(n_components_80, n_components_kaiser))
        print("\nRecommended number of components:")
        print(f"- For 80% variance: {n_components_80}")
        print(f"- Kaiser criterion (eigenvalue > 1): {n_components_kaiser}")
        print(f"- Selected: {n_components}")

        # Fit PCA with selected components
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(data)

        # PCA Loading analysis
        print("\n5.1.1 PCA Loading Analysis:")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=data.columns,
        )
        print("PCA Loadings:")
        print(loadings.round(3))

        # Identify top contributing variables for PC1, PC2, PC3
        for i in range(min(3, n_components)):
            top_vars = loadings.iloc[:, i].abs().sort_values(ascending=False).head(3)
            print(f"Top contributors to PC{i + 1}: {', '.join(top_vars.index)}")

        # Detailed loading plots for each component
        fig, axes = plt.subplots(
            1, min(n_components, 3), figsize=(5 * min(n_components, 3), 5)
        )
        if n_components == 1:
            axes = [axes]
        elif n_components == 2:
            axes = axes if hasattr(axes, "__len__") else [axes]
        for i in range(min(n_components, 3)):
            loadings_pc = loadings.iloc[:, i].sort_values(key=np.abs, ascending=False)
            axes[i].barh(loadings_pc.index, loadings_pc.values)
            axes[i].set_title(f"Loadings for PC{i + 1}")
            axes[i].axvline(0, color="k", linestyle="--")
        plt.tight_layout()
        plt.show()

        # 5.2 PCA Visualization and Analysis
        print("\n5.2 PCA Visualization and Analysis:")
        pca_df = pd.DataFrame(
            pca_transformed,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=data.index,
        )
        # Merge with original df for labels/colors
        merged = self.df.loc[data.index].copy()
        pca_df = pd.concat([pca_df, merged.reset_index(drop=True)], axis=1)

        # Scatter plot PC1 vs PC2, color by ln_gdp/hdi/year if available
        for color_col in color_cols:
            if color_col in pca_df.columns:
                plt.figure(figsize=(8, 6))
                sc = plt.scatter(
                    pca_df["PC1"],
                    pca_df["PC2"],
                    c=pca_df[color_col],
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.colorbar(sc, label=color_col)
                if label_col in pca_df.columns:
                    for i, txt in enumerate(pca_df[label_col]):
                        if i % max(1, len(pca_df) // 30) == 0:  # avoid clutter
                            plt.annotate(
                                txt,
                                (pca_df["PC1"].iloc[i], pca_df["PC2"].iloc[i]),
                                fontsize=8,
                                alpha=0.7,
                            )
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title(f"PCA: PC1 vs PC2 colored by {color_col}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        # 3D scatter plot if n_components >= 3
        if n_components >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pca_df["PC1"],
                pca_df["PC2"],
                pca_df["PC3"],
                c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
                cmap="viridis",
                alpha=0.7,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            plt.title("PCA: PC1 vs PC2 vs PC3")
            plt.tight_layout()
            plt.show()

        # PCA biplot (PC1 vs PC2 with arrows for variables)
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.5)
            for i, var in enumerate(data.columns):
                plt.arrow(
                    0,
                    0,
                    loadings.iloc[i, 0] * 3,
                    loadings.iloc[i, 1] * 3,
                    color="r",
                    alpha=0.7,
                    head_width=0.05,
                )
                plt.text(
                    loadings.iloc[i, 0] * 3.2,
                    loadings.iloc[i, 1] * 3.2,
                    var,
                    color="r",
                    fontsize=10,
                )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA Biplot (PC1 vs PC2)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Time evolution plots: track countries' movement in PCA space
        if "year" in pca_df.columns and label_col in pca_df.columns:
            plt.figure(figsize=(12, 8))
            for country in pca_df[label_col].unique():
                country_data = pca_df[pca_df[label_col] == country].sort_values("year")
                plt.plot(
                    country_data["PC1"],
                    country_data["PC2"],
                    marker="o",
                    label=country,
                    alpha=0.5,
                )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Country Trajectories in PCA Space (PC1 vs PC2)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # PCA stability analysis: compare PCA results across different time periods
        if "year" in pca_df.columns:
            years = sorted(pca_df["year"].unique())
            if len(years) > 1:
                first_year = years[0]
                last_year = years[-1]
                for yr in [first_year, last_year]:
                    idx = pca_df["year"] == yr
                    plt.scatter(
                        pca_df.loc[idx, "PC1"],
                        pca_df.loc[idx, "PC2"],
                        label=f"Year {yr}",
                        alpha=0.7,
                    )
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title("PCA: PC1 vs PC2 by Year")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        # Contribution of variables plot (contribution to each PC)
        contrib = loadings.abs().div(loadings.abs().sum(axis=0), axis=1)
        contrib.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
        plt.title("Variable Contributions to Each Principal Component")
        plt.ylabel("Contribution")
        plt.xlabel("Variable")
        plt.tight_layout()
        plt.show()

        # 5.3 Alternative Dimensionality Reduction
        print("\n5.3 Alternative Dimensionality Reduction:")

        # t-SNE with different perplexities
        perplexities = [5, 10, 30, 50]
        for perp in perplexities:
            try:
                tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
                tsne_result = tsne.fit_transform(data)
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    tsne_result[:, 0],
                    tsne_result[:, 1],
                    c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
                    cmap="viridis",
                    alpha=0.7,
                )
                plt.title(f"t-SNE (perplexity={perp})")
                plt.xlabel("t-SNE 1")
                plt.ylabel("t-SNE 2")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"t-SNE failed for perplexity={perp}: {e}")

        # UMAP (if available)
        try:
            reducer = umap.UMAP(random_state=42)
            umap_result = reducer.fit_transform(data)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                umap_result[:, 0],
                umap_result[:, 1],
                c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
                cmap="viridis",
                alpha=0.7,
            )
            plt.title("UMAP")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("UMAP is not installed. Skipping UMAP visualization.")

        # Factor Analysis
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        fa_result = fa.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            fa_result[:, 0],
            fa_result[:, 1],
            c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
            cmap="viridis",
            alpha=0.7,
        )
        plt.title("Factor Analysis (FA1 vs FA2)")
        plt.xlabel("FA1")
        plt.ylabel("FA2")
        plt.tight_layout()
        plt.show()

        # ICA
        ica = FastICA(n_components=n_components, random_state=42)
        try:
            ica_result = ica.fit_transform(data)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                ica_result[:, 0],
                ica_result[:, 1],
                c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
                cmap="viridis",
                alpha=0.7,
            )
            plt.title("Independent Component Analysis (IC1 vs IC2)")
            plt.xlabel("IC1")
            plt.ylabel("IC2")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"ICA failed: {e}")

        # MDS
        try:
            mds = MDS(n_components=2, random_state=42)
            mds_result = mds.fit_transform(data)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                mds_result[:, 0],
                mds_result[:, 1],
                c=pca_df[color_cols[0]] if color_cols[0] in pca_df.columns else "b",
                cmap="viridis",
                alpha=0.7,
            )
            plt.title("Multidimensional Scaling (MDS)")
            plt.xlabel("MDS1")
            plt.ylabel("MDS2")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"MDS failed: {e}")

        return {
            "pca": pca,
            "pca_transformed": pca_transformed,
            "loadings": loadings,
            "pca_df": pca_df,
        }


if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv(
        r"C:\Users\VAQ\OneDrive - TRƯỜNG ĐẠI HỌC MỞ TP.HCM\University\Research paper\RP_2 ngươi bạn\Menthology\Machine Learning Method\Data_csv\processed_dataset.csv"
    )

    # Initialize the ICTAnalyzer with the dataframe
    analyzer = ICTAnalyzer(df=df)

    # Run the analysis methods
    desc_stats, skew_kurt_df, outlier_summary = analyzer.basic_info()
    analyzer.missing_data_analysis()
    analyzer.time_series_analysis()
    correlations, vif_df = analyzer.correlation_analysis()
    analyzer.feature_engineering()
    analyzer.feature_selection(target_var="gdp")  # Specify your target variable
    scaled_data = analyzer.scaling_analysis()
    if scaled_data and "StandardScaler" in scaled_data:
        analyzer.distribution_analysis(
            scaled_data["StandardScaler"],
            scaled_data["StandardScaler"].columns.tolist(),
        )
    analyzer.pca_analysis()
    print("Analysis completed successfully.")
