import dagster as dg

from bangla_ocr.defs import defs


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
