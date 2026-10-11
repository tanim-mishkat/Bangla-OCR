import dagster as dg

from bangla_ocr import checks, split
from bangla_ocr.defs import defs, splits, splits_contract


def test_definitions_load():
    dg.Definitions.validate_loadable(defs)


def test_asset_graph():
    graph = defs.resolve_asset_graph()
    deps = {
        key.to_user_string(): sorted(p.to_user_string() for p in graph.get(key).parent_keys)
        for key in graph.get_all_asset_keys()
    }
    assert deps == {
        "raw_sources": [],
        "reid_crops": ["raw_sources"],
        "text_corpus": ["raw_sources"],
        "synth_images": ["raw_sources", "text_corpus"],
        "ingested": ["reid_crops", "synth_images"],
        "cleaned": ["ingested"],
        "splits": ["cleaned"],
    }


def test_splits_contract_is_blocking():
    specs = list(splits_contract.check_specs)
    assert tuple(s.name for s in specs) == checks.NAMES
    assert all(s.blocking and s.asset_key.to_user_string() == "splits" for s in specs)


def test_failed_check_blocks_downstream(monkeypatch):
    monkeypatch.setattr(split, "split", lambda: {"test": {"reid2019": 1}})  # skip real I/O
    monkeypatch.setattr(
        checks,
        "run",
        lambda: {n: ["leak"] if n == "reid_only_in_test" else [] for n in checks.NAMES},
    )

    @dg.asset(deps=[splits])
    def publish_stub() -> None: ...

    result = dg.materialize(
        [splits, publish_stub, splits_contract],
        raise_on_error=False,
        instance=dg.DagsterInstance.ephemeral(),
    )

    failed = {e.check_name for e in result.get_asset_check_evaluations() if not e.passed}
    assert failed == {"reid_only_in_test"}
    assert result.asset_materializations_for_node("splits")
    assert not result.asset_materializations_for_node("publish_stub")
