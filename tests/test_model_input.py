import polars as pl

from src.models import preprocessing as model_input


def test_get_feature_columns_keeps_only_numeric_and_boolean():
    schema = {
        "terminal_code": pl.Int32,
        "timestamp": pl.Datetime,
        "date": pl.Date,
        "is_weekend": pl.Boolean,
        "is_training_origin": pl.Boolean,
        "address": pl.Utf8,
        "zone_desc": pl.Categorical,
        "hour": pl.Int8,
        "occupancy_rate": pl.Float64,
        "is_free": pl.Boolean,
        "terminal_id": pl.Int64,
        "y_h1h": pl.Float64,
        "pred_current": pl.Float64,
    }

    got = model_input.get_feature_columns(schema)

    assert got == ["hour", "occupancy_rate", "is_free", "terminal_id"]


def test_encode_terminal_codes_maps_only_training_terminals():
    train = pl.DataFrame({"terminal_code": [300, 100, 200, 100]})
    valid = pl.DataFrame({"terminal_code": [100, 999]})
    test = pl.DataFrame({"terminal_code": [200]})

    train_enc, valid_enc, test_enc, terminal_map = model_input.encode_terminal_codes(
        train, valid, test
    )

    assert terminal_map["terminal_code"].to_list() == [100, 200, 300]
    assert terminal_map["terminal_id"].to_list() == [0, 1, 2]
    assert terminal_map.schema["terminal_id"] == pl.Int64

    # train keeps every row, unseen terminals are dropped from valid and test
    assert train_enc.height == 4
    assert sorted(train_enc["terminal_id"].unique().to_list()) == [0, 1, 2]
    assert valid_enc["terminal_code"].to_list() == [100]
    assert valid_enc["terminal_id"].to_list() == [0]
    assert test_enc["terminal_id"].to_list() == [1]


def test_encode_terminal_codes_keeps_existing_columns():
    train = pl.DataFrame({"terminal_code": [100], "occupancy_rate": [0.5]})

    train_enc, _, _, _ = model_input.encode_terminal_codes(train, train, train)

    assert set(train_enc.columns) == {"terminal_code", "occupancy_rate", "terminal_id"}
