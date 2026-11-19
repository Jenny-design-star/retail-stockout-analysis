import numpy as np
import pandas as pd
import pytest
from allocation import allocate, policy_metrics, paired_bootstrap


def sample():
    return pd.DataFrame(
        {
            "dt": pd.to_datetime(["2024-01-01"] * 6 + ["2024-01-02"] * 6),
            "store_id": [1, 1, 1, 2, 2, 3] * 2,
            "product_id": [10, 11, 12, 10, 11, 10] * 2,
            "score": [0.99, 0.98, 0.97, 0.6, 0.5, 0.1] * 2,
            "label": [1, 1, 0, 1, 0, 0] * 2,
        }
    )


@pytest.mark.parametrize("k", [1, 2, 3, 10])
def test_budgets_and_cap(k):
    x = sample()
    f = allocate(x, k)
    for _, idx in x.groupby("dt").groups.items():
        expected = x.loc[idx].groupby("store_id").size().clip(upper=k).sum()
        assert (f.loc[idx].sum() == expected).all()
        assert (
            x.loc[idx].loc[f.loc[idx, "one_per_store_then_global"]].store_id.nunique()
            == 3
        )
    assert (x.loc[f.per_store_top_n].groupby(["dt", "store_id"]).size() <= k).all()


def test_known_lists_and_precision():
    x = sample()
    f = allocate(x, 1)
    assert list(np.flatnonzero(f.global_matched)) == [0, 1, 2, 6, 7, 8]
    assert list(np.flatnonzero(f.per_store_top_n)) == [0, 3, 5, 6, 9, 11]
    assert f.per_store_top_n.equals(f.one_per_store_then_global)
    d, w = policy_metrics(x, f, 1)
    assert np.allclose(d.precision, 2 / 3)
    ci, diff = paired_bootstrap(w, 100, 42)
    assert len(ci) == 9 and len(diff) == 6
    assert np.isfinite(ci[["lo", "hi"]]).all().all()
    assert (ci.n_valid <= 100).all() and (ci.n_valid > 0).all()
    pd.testing.assert_frame_equal(ci, paired_bootstrap(w, 100, 42)[0])


def test_independent_of_labels_and_input_order_with_ties():
    x = sample()
    x["score"] = 0.5
    f = allocate(x, 2)
    y = x.sample(frac=1, random_state=9).reset_index(drop=True)
    y.label = 1 - y.label
    z = (
        pd.concat([y[["dt", "store_id", "product_id"]], allocate(y, 2)], axis=1)
        .set_index(["dt", "store_id", "product_id"])
        .sort_index()
    )
    expected = (
        pd.concat([x[["dt", "store_id", "product_id"]], f], axis=1)
        .set_index(["dt", "store_id", "product_id"])
        .sort_index()
    )
    pd.testing.assert_frame_equal(z, expected)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_bad_k(k):
    with pytest.raises(ValueError):
        allocate(sample(), k)


def test_invalid_data():
    x = sample()
    x.loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        allocate(x, 1)
    with pytest.raises(ValueError):
        allocate(pd.concat([sample(), sample()]), 1)
