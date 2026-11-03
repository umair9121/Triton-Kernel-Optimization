import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def probe_results():
    from tritonlens.validate import run_probes

    return run_probes()


def _by_kind(results, kind):
    selected = [r for r in results if r.layout_kind == kind]
    assert selected, f"no probe produced a {kind} layout"
    return selected


def test_blocked_layouts_match_gpu(probe_results):
    for result in _by_kind(probe_results, "blocked"):
        assert result.ok, f"{result.name}: {result.mismatches} mismatching elements"


def test_slice_layouts_match_gpu(probe_results):
    for result in _by_kind(probe_results, "slice"):
        assert result.ok, f"{result.name}: {result.mismatches} mismatching elements"


def test_mma_v2_layouts_match_gpu(probe_results):
    for result in _by_kind(probe_results, "nvidia_mma"):
        assert result.ok, f"{result.name}: {result.mismatches} mismatching elements"


def test_broadcast_probe_is_replicated(probe_results):
    small = [r for r in probe_results if r.name.startswith("1d 32 ")]
    assert small and small[0].replication == 4
