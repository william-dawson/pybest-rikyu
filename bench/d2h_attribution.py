"""Attribute every device-to-host transfer to the line of PyBEST that asked for it.

Why this matters. crosslib_batching consumes a transfer in two different ways:

    result[...] += move_tensor_to_cpu(part)      lines 437, 489, 998, 1062, 1487
    result[a:b, c:d, :] = move_tensor_to_cpu(..) lines 541, 1117
    result = move_tensor_to_cpu(result_cp)       lines 351, 582, 925, 1158

The first two already know the destination, so the DMA could write it directly --
`t.get(out=destination)` -- with no temporary and no copy. The third RETURNS the
array, so it escapes the function and a view into a reused buffer would be
corrupted by the next call; that is why our pinned patch has to copy.

Job 161947 priced the difference: 10.84 GB/s as PyBEST plus our patch does it,
against 192.97 GB/s for the same DMA writing a registered destination. An 18x
gap, invisible to nsys because the copy is host-side work.

So the value of recommending `out=` upstream is set by how much of the transfer
VOLUME flows through the sites that could use it. This measures exactly that, by
recording the caller's line number for every transfer. No timing, no GPU
dependence beyond running at all -- it is pure attribution.

Enable with PYBEST_D2H_ATTRIB=1. The table prints at exit.
"""
from __future__ import annotations

import atexit
import collections
import os
import sys


def install(verbose: bool = True) -> bool:
    try:
        import pybest.linalg.crosslib_batching as _cb
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# d2h_attrib: not installed ({exc})", flush=True)
        return False

    orig = _cb.move_tensor_to_cpu
    # line -> [count, bytes]
    stats: dict[tuple[str, int], list[int]] = collections.defaultdict(
        lambda: [0, 0])

    def counted(tensor):
        f = sys._getframe(1)
        key = (f.f_code.co_name, f.f_lineno)
        rec = stats[key]
        rec[0] += 1
        try:
            n = tensor.nbytes
        except AttributeError:
            try:                                            # torch
                n = tensor.element_size() * tensor.numel()
            except Exception:                               # noqa: BLE001
                n = 0
        rec[1] += n
        return orig(tensor)

    _cb.move_tensor_to_cpu = counted

    # Which call sites could pass out= because they already know the destination.
    CAN_USE_OUT = {437, 489, 998, 1062, 1487, 541, 1117}
    ESCAPES = {351, 582, 925, 1158}

    def report() -> None:
        if not stats:
            print("# d2h_attrib: no transfers recorded", flush=True)
            return
        tot_n = sum(v[0] for v in stats.values())
        tot_b = sum(v[1] for v in stats.values())
        print("\n# d2h attribution: every move_tensor_to_cpu by caller line",
              flush=True)
        print(f"#   {'function':<34}{'line':>6}{'calls':>8}{'MB':>12}"
              f"{'% bytes':>9}  kind", flush=True)
        for (name, line), (cnt, nb) in sorted(
                stats.items(), key=lambda kv: -kv[1][1]):
            kind = ("can use out=" if line in CAN_USE_OUT else
                    "escapes" if line in ESCAPES else "unclassified")
            print(f"#   {name:<34}{line:>6}{cnt:>8}{nb / 1e6:>12.1f}"
                  f"{100 * nb / max(tot_b, 1):>8.1f}%  {kind}", flush=True)
        out_b = sum(v[1] for k, v in stats.items() if k[1] in CAN_USE_OUT)
        esc_b = sum(v[1] for k, v in stats.items() if k[1] in ESCAPES)
        unk_b = tot_b - out_b - esc_b
        print(f"#   {'TOTAL':<34}{'':>6}{tot_n:>8}{tot_b / 1e6:>12.1f}"
              f"{100.0:>8.1f}%", flush=True)
        print(f"# d2h_can_use_out_pct={100 * out_b / max(tot_b, 1):.1f} "
              f"escapes_pct={100 * esc_b / max(tot_b, 1):.1f} "
              f"unclassified_pct={100 * unk_b / max(tot_b, 1):.1f}", flush=True)
        print("# The first number is the share of D2H volume that could go "
              "straight to its\n# destination at 193 GB/s instead of through a "
              "staging buffer at 10.8 GB/s.", flush=True)

    atexit.register(report)
    if verbose:
        print("# d2h_attrib: counting move_tensor_to_cpu by caller line",
              flush=True)
    return True


if os.environ.get("PYBEST_D2H_ATTRIB") == "1" and __name__ != "__main__":
    pass   # the driver calls install() explicitly; importing does nothing
