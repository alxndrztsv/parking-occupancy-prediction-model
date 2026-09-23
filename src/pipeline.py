import polars as pl

from src import config
from src.data import clean, enrich, ingest, normalize, validate
from src.features import features, grid, split
from src.models import baselines, metrics, train_models

ENRICHED_PATH = config.ENRICHED_DIR / "clean.parquet"
OPERATIONAL_PATH = config.PROCESSED_DIR / "operational_table.parquet"
CONTINUOUS_PATH = config.PROCESSED_DIR / "continuous_table.parquet"
FEATURES_PATH = config.PROCESSED_DIR / "features.parquet"

TRAIN_PATH = config.SPLITS_DIR / "train.parquet"
VALID_PATH = config.SPLITS_DIR / "valid.parquet"
TEST_PATH = config.SPLITS_DIR / "test.parquet"

BASELINE_METRICS_PATH = config.REPORTS_DIR / "baseline_metrics.csv"
TARGET_PROFILE_PATH = config.REPORTS_DIR / "target_profile.csv"
MODEL_COMPARISON_PATH = config.REPORTS_DIR / "model_comparison.csv"


def make_directories() -> None:
    config.NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    config.ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_stage_normalize() -> None:
    # Normalize
    normalize.normalize_raw_dir()


def run_stage_loading() -> None:
    # Ingest
    lf = ingest.scan_normalized_files(config.NORMALIZED_DIR)
    normalized_df = lf.collect()    
    validate.validate_normalized(normalized_df)

    # Clean
    clean_df = clean.clean(normalized_df)

    # Enrich
    enriched_df = enrich.enrich(clean_df)
    validate.validate_enriched(enriched_df)
    enriched_df.write_parquet(ENRICHED_PATH)


def run_stage_grid() -> None:
    enriched_df = pl.read_parquet(ENRICHED_PATH)

    # Build initial occupancy table
    operational_lf = grid.build_operational_occupancy_table(enriched_df)
    operational_df = operational_lf.collect()
    operational_df.write_parquet(OPERATIONAL_PATH)

    # Build continuous 24-hour grid
    continuous_lf = grid.build_continuous_occupancy_table(operational_df)
    continuous_df = continuous_lf.collect()
    continuous_df.write_parquet(CONTINUOUS_PATH)


def run_stage_features() -> None:
    continuous_df = pl.read_parquet(CONTINUOUS_PATH)

    # Build feature table
    features_lf = features.build_feature_table(continuous_df)
    features_df = features_lf.collect()
    features_df.write_parquet(FEATURES_PATH)


def run_stage_split() -> None:
    features_df = pl.read_parquet(FEATURES_PATH)

    # Filter valid rows and split by date
    valid_rows_lf = split.filter_valid_rows(features_df)
    train_lf, valid_lf, test_lf = split.split_by_date(valid_rows_lf)

    train_df = train_lf.collect()
    valid_df = valid_lf.collect()
    test_df = test_lf.collect()

    train_df.write_parquet(TRAIN_PATH)
    valid_df.write_parquet(VALID_PATH)
    test_df.write_parquet(TEST_PATH)


def run_stage_baseline() -> tuple[pl.DataFrame, pl.DataFrame]:
    train_df = pl.read_parquet(TRAIN_PATH)
    valid_df = pl.read_parquet(VALID_PATH)
    test_df = pl.read_parquet(TEST_PATH)

    # Describe targets and evaluate baselines on the validation and test splits
    baseline_metrics_list = []
    target_profile_list = []

    for horizon_name in config.HORIZONS_MINUTES:
        target = f"y_{horizon_name}"
        mase_scale = metrics.naive_scale(train_df, target)

        baseline_metrics_list.append(
            baselines.evaluate_baselines_by_split(
                splits={"valid": valid_df, "test": test_df},
                target=target,
                horizon_name=horizon_name,
                mase_scale=mase_scale,
            )
        )

        for split_name, split_df in (
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ):
            target_profile_list.append(
                {
                    "horizon": horizon_name,
                    "split": split_name,
                    **metrics.target_profile(split_df, target, mase_scale),
                }
            )

    baseline_metrics = pl.concat(baseline_metrics_list)
    baseline_metrics.write_csv(config.REPORTS_DIR / "baseline_metrics.csv")

    target_profiles = pl.DataFrame(target_profile_list)
    target_profiles.write_csv(config.REPORTS_DIR / "target_profile.csv")
    print("\n--- Target Profile ---")
    print(target_profiles)
    print("\n--- Baseline Metrics ---")
    print(baseline_metrics)

    return baseline_metrics, target_profiles


def run_stage_train() -> pl.DataFrame:
    train_df = pl.read_parquet(TRAIN_PATH)
    valid_df = pl.read_parquet(VALID_PATH)
    test_df = pl.read_parquet(TEST_PATH)

    # Train models
    model_metrics = train_models.train_all_models(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        model_dir=config.MODELS_DIR,
        report_dir=config.REPORTS_DIR,
    )
    print("\n--- Model Metrics ---")
    print(model_metrics)

    return model_metrics


def run_stage_compare(
    model_metrics: pl.DataFrame,
    baseline_metrics: pl.DataFrame,
    target_profiles: pl.DataFrame,
) -> None:
    # Compare models against baselines on identical metrics
    comparison = metrics.build_comparison_table(model_metrics, baseline_metrics)

    comparison = comparison.join(
        target_profiles.select("horizon", "split", "zero_share"),
        on=["horizon", "split"],
        how="left",
    )
    comparison.write_csv(MODEL_COMPARISON_PATH)
    print("\n--- Model vs Baseline ---")
    print(comparison)
    

def main():
    make_directories()
    run_stage_normalize()
    run_stage_loading()
    run_stage_grid()
    run_stage_features()
    run_stage_split()

    baseline_metrics, target_profiles = run_stage_baseline()
    model_metrics = run_stage_train()

    run_stage_compare(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        target_profiles=target_profiles,
    )

    print("Pipeline complete. Artifacts saved to data/, models/, and reports/")


if __name__ == "__main__":
    main()