def test_package_imports():
    import traceotter

    assert hasattr(traceotter, "get_client")
