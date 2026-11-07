
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


def train_and_save_model():
    # --------------------------------------------------
    # 1. Set project paths
    # --------------------------------------------------
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    model_dir = os.path.join(project_dir, "backend", "model")

    mineral_path = os.path.join(data_dir, "Minerals.csv")
    commodity_path = os.path.join(
        data_dir, "world_mining_commodities_clean.csv"
    )
    companies_path = os.path.join(
        data_dir, "116_world_mining_companies_clean.csv"
    )
    excel_path = os.path.join(data_dir, "commodity_info.xlsx")

    # --------------------------------------------------
    # 2. Load mineral dataset
    # --------------------------------------------------
    if not os.path.exists(mineral_path):
        raise FileNotFoundError(
            f"Minerals.csv not found at: {mineral_path}"
        )

    df = pd.read_csv(mineral_path)
    df.columns = df.columns.str.strip()

    print("\nMinerals dataset loaded.")
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))

    if "MohsHardness" not in df.columns:
        raise ValueError(
            "The target column 'MohsHardness' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Standardize the geographic key only if one exists.
    if "Region" not in df.columns and "Country" in df.columns:
        df.rename(columns={"Country": "Region"}, inplace=True)

    # --------------------------------------------------
    # 3. Load commodity dataset and calculate diversity
    # --------------------------------------------------
    if os.path.exists(commodity_path):
        df_comm = pd.read_csv(commodity_path)
        df_comm.columns = df_comm.columns.str.strip()

        country_col = next(
            (c for c in ["Country", "Region", "country", "region"]
             if c in df_comm.columns),
            None
        )

        commodity_col = next(
            (c for c in ["Commodity", "Mineral", "commodity", "mineral"]
             if c in df_comm.columns),
            None
        )

        if country_col and commodity_col and "Region" in df.columns:
            region_diversity = (
                df_comm.groupby(country_col)[commodity_col]
                .nunique()
                .reset_index()
            )

            region_diversity.columns = [
                "Region", "CommodityDiversityIndex"
            ]

            df = df.merge(
                region_diversity,
                on="Region",
                how="left"
            )

            print("Commodity diversity merged successfully.")

        else:
            print(
                "Commodity merge skipped: compatible geographic "
                "and commodity columns were not found."
            )
    else:
        print("Commodity dataset not found.")

    if "CommodityDiversityIndex" not in df.columns:
        df["CommodityDiversityIndex"] = float("nan")

    # --------------------------------------------------
    # 4. Load mining company dataset
    # --------------------------------------------------
    if os.path.exists(companies_path):
        df_comp = pd.read_csv(companies_path)
        df_comp.columns = df_comp.columns.str.strip()

        company_country_col = next(
            (c for c in ["Country", "Region", "country", "region"]
             if c in df_comp.columns),
            None
        )

        if company_country_col and "Region" in df.columns:
            comp_metrics = (
                df_comp.groupby(company_country_col)
                .size()
                .reset_index(name="ActiveMiningCompanies")
            )

            comp_metrics.columns = [
                "Region", "ActiveMiningCompanies"
            ]

            df = df.merge(
                comp_metrics,
                on="Region",
                how="left"
            )

            print("Mining company metrics merged successfully.")

        else:
            print(
                "Company merge skipped: compatible geographic "
                "columns were not found."
            )
    else:
        print("Mining companies dataset not found.")

    if "ActiveMiningCompanies" not in df.columns:
        df["ActiveMiningCompanies"] = float("nan")

    # --------------------------------------------------
    # 5. Load commodity information Excel file
    # --------------------------------------------------
    if os.path.exists(excel_path):
        df_info = pd.read_excel(excel_path)

        print("Commodity information Excel loaded.")
        print("Excel columns:", df_info.columns.tolist())

        # Do not invent economic values or assume a join key.
        # A real Excel merge requires verified matching columns.
        if "EconomicValueIndex" not in df.columns:
            df["EconomicValueIndex"] = float("nan")
    else:
        print("Commodity information Excel file not found.")

        if "EconomicValueIndex" not in df.columns:
            df["EconomicValueIndex"] = float("nan")

    # --------------------------------------------------
    # 6. Prepare model features and target
    # --------------------------------------------------
    feature_cols = [
        "CalculatedDensity",
        "SpecificGravity",
        "ChemicalPurity",
        "RefractiveIndex",
        "CrystalSystem",
        "Region",
        "CommodityDiversityIndex",
        "ActiveMiningCompanies",
        "EconomicValueIndex",
    ]

    available_features = [
        col for col in feature_cols if col in df.columns
    ]

    if not available_features:
        raise ValueError(
            "No supported model features were found in Minerals.csv. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["MohsHardness"] = pd.to_numeric(
        df["MohsHardness"], errors="coerce"
    )

    df = df.dropna(subset=["MohsHardness"]).copy()

    X = df[available_features].copy()
    y = df["MohsHardness"].copy()

    # Remove rows with infinite target values.
    valid_target = y.notna() & y.map(
        lambda value: abs(value) != float("inf")
    )
    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    if len(X) < 5:
        raise ValueError(
            f"Only {len(X)} usable rows remain. "
            "At least 5 rows are required for this training setup."
        )

    # Ensure numeric-looking fields are handled consistently.
    numeric_candidates = [
        "CalculatedDensity",
        "SpecificGravity",
        "ChemicalPurity",
        "RefractiveIndex",
        "CommodityDiversityIndex",
        "ActiveMiningCompanies",
        "EconomicValueIndex",
    ]

    for col in numeric_candidates:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    num_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cat_features = [
        col for col in X.columns if col not in num_features
    ]

    transformers = []

    if num_features:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("num", num_transformer, num_features)
        )

    if cat_features:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers.append(
            ("cat", cat_transformer, cat_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------
    # 7. Build XGBoost regression pipeline
    # --------------------------------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror",
        )),
    ])

    # --------------------------------------------------
    # 8. Split, train and evaluate
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n========== MODEL RESULTS ==========")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features used: {available_features}")
    print(f"R² score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    # --------------------------------------------------
    # 9. Save trained model
    # --------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir, "mineral_xgb_model.pkl"
    )

    joblib.dump(model, model_path)

    print("\nModel trained and saved successfully.")
    print(f"Saved at: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
