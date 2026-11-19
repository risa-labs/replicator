"""What a 3-axis mill would leave of each part of a BuildProgram: stock, setups, and what stays uncut.

    .venv/bin/python -m tools.machinability replicator/fixtures/raspberry_pi_5.json
    .venv/bin/python -m tools.machinability part.json --machine makera_z1 --tool em_6 --max-setups 2

Each part is measured alone, as it stands in the build (core/machining.py). Exits 1 when a program is
invalid or any part has a `fix` issue.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from replicator.core import machining
from replicator.core.evaluate import EvalError, Evaluator, RawMesh
from replicator.core.program import ProgramError, check


def report(path: Path, machine: machining.Machine, tool: machining.Tool, max_setups: int, allowance: float) -> bool:
    try:
        program = check(json.loads(path.read_text()))
        shapes = Evaluator(path.parent).step(program)
    except (OSError, json.JSONDecodeError, ProgramError, EvalError) as e:
        print(f"{path}: INVALID: {e}")
        return False
    units = program.get("units", "mm")
    tools = tuple(machining.TOOLS.values())
    print(f"{path}: {len(shapes)} part{'s' if len(shapes) != 1 else ''} on the {machine.vendor} {machine.name} with "
          f"the {tool.name} (reach {tool.reach:g} mm), at most {max_setups} setups")
    ok = True
    for s in shapes:
        if isinstance(s.manifold, RawMesh):
            print(f"  {s.label}: an open surface, not measured")
            continue
        t0 = time.perf_counter()
        try:
            r = machining.analyse(s.manifold, machine, tool, units=units, part=s.id, label=s.label,
                                  max_setups=max_setups, allowance=allowance, tools=tools)
        except machining.MachiningError as e:
            print(f"  {s.label}: NOT MEASURED: {e}")
            ok = False
            continue
        ms = (time.perf_counter() - t0) * 1000
        stock = " x ".join(f"{v:.0f}" for v in r.stock)
        if not r.fits:
            line = "does not fit"
        elif r.to_remove <= machining.NEGLIGIBLE:
            line = "a plain block, nothing to cut"
        else:
            left = "nothing left" if r.leftover <= machining.NEGLIGIBLE else \
                f"{r.leftover:.1f} mm^3 left (hidden {r.undercut:.1f}, too tight {r.corner:.1f}, too deep {r.deep:.1f})"
            line = (f"{r.to_remove / (r.to_remove + r.volume):.0%} cut away, {len(r.setups)} "
                    f"setup{'s' if len(r.setups) != 1 else ''} ({', '.join(r.setups)}), {left}")
        print(f"  {s.label}: stock {stock} mm, {line} [{ms:.0f} ms]")
        for i in machining.explain(r, machine, tool, tools, s.component):
            print(f"    {i.severity.upper():4} {i.kind:13} {i.message}")
            ok = ok and i.severity != "fix"
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("programs", nargs="+", type=Path)
    ap.add_argument("--machine", default="makera_z1", choices=sorted(machining.MACHINES))
    ap.add_argument("--tool", default="", choices=["", *sorted(machining.TOOLS)], help="default: the machine's standard collet")
    ap.add_argument("--max-setups", type=int, default=3)
    ap.add_argument("--allowance", type=float, default=0.0, help="stock left on every face of the bounding box, mm")
    a = ap.parse_args()
    mach = machining.MACHINES[a.machine]
    cutter = machining.TOOLS[a.tool or machining.DEFAULT_TOOL[a.machine]]
    done = [report(p, mach, cutter, a.max_setups, a.allowance) for p in a.programs]
    sys.exit(0 if all(done) else 1)
