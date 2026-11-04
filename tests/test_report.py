import json
from pathlib import Path

from tritonlens.analyze import analyze_captured
from tritonlens.capture import CapturedKernel
from tritonlens.report import build_report, render_html

FIXTURES = Path(__file__).parent / "fixtures"


def report_for(name):
    return build_report(analyze_captured(CapturedKernel.from_files(FIXTURES / f"{name}.ttgir")))


def test_report_is_json_serializable():
    for name in ["matmul_kernel", "softmax_kernel", "transpose_tiled_kernel"]:
        data = report_for(name)
        json.dumps(data)
        assert data["kernel"]["name"] == name


def test_report_contents_matmul():
    data = report_for("matmul_kernel")
    kinds = {t["layout"]["kind"] for t in data["tensors"]}
    assert {"blocked", "dot_op", "nvidia_mma"} <= kinds
    for t in data["tensors"]:
        n = 1
        for s in t["shape"]:
            n *= s
        assert len(t["warp"]) == len(t["lane"]) == len(t["reg"]) == n
    buffers = {b["id"]: b for b in data["shared"]["buffers"]}
    assert len(buffers) == 3
    tile = next(b for b in buffers.values() if not b.get("estimated"))
    assert len(tile["banks"]) == tile["tile"][0]
    assert all(a["bufferId"] in buffers for a in data["shared"]["accesses"])
    assert data["tiling"]["tileShape"] == [64, 64]
    assert data["source"]["lines"] is not None
    perf = data["perf"]
    assert perf["occupancy"]["blocks_per_sm"] == 5
    assert perf["waves"]["total_blocks"] == 256
    assert perf["roofline"]["flops"] == 2 * 1024 ** 3
    assert perf["budget"]["tensorBytes"] == 3 * 1024 * 1024 * 2
    assert perf["budget"]["l2Fits"] is True
    assert perf["advice"] and perf["advice"][0]["current"] is True


def test_html_is_self_contained():
    html = render_html(report_for("softmax_kernel"))
    assert "<script src" not in html and "<link" not in html
    assert "__DATA__" not in html and "__JS__" not in html
    start = html.index('id="tl-data">') + len('id="tl-data">')
    end = html.index("</script>", start)
    payload = json.loads(html[start:end].replace("<\\/", "</"))
    assert payload["kernel"]["name"] == "softmax_kernel"
