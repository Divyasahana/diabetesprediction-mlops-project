import importlib


def test_import_src_modules():
    modules = [
        "src.train",
        "src.preprocess",
        "src.evaluate",
        "src.utils",
    ]

    for module in modules:
        importlib.import_module(module)

    assert True
