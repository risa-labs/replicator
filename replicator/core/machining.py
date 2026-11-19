"""Machinability on a 3-axis mill: can a flat end mill reach everything that has to be cut away, and
from how many setups? (docs/REQUIREMENTS-workbench.md, 5.11, the CNC pack: tool access, stock.)

A printer adds material, so a solid that fits the bed can be made. A mill takes material away from a
block, and the tool only comes down its own axis: what it cannot see from above, cannot fit into, or
cannot reach down to stays in the block. This module measures that remainder on the built solid.

How it is measured. A *setup* is the stock clamped with one of its six faces up. In a setup the part is
cut into horizontal layers, top first. On each layer:

  need   what must become air: the plane minus the part's section
  free   what the tool can see: the plane minus the shadow of everything above (the union of the
         sections so far). need minus free is an undercut: material the part hides from above.
  reach  where a cutter of the tool's radius fits: free opened by a disc (shrunk by the radius, grown
         back). free minus reach is an inside corner sharper than the tool, or a slot or hole narrower
         than it. Below the tool's reach nothing is cut at all.

Several setups leave only what every one of them leaves, so the remainder of a set of setups is the
intersection of their remainders, and the fewest setups whose remainder is empty is the answer. A face
and the one opposite share their layers, so that pair is intersected layer by layer in the plane and
is exact for a prismatic part; setups about different axes are extruded back into solids and
intersected there. When no set clears the part, the answer is the set that leaves least without
re-clamping for little (_Part.search). The remainder is reported in cubic millimetres, split by
cause, with where the largest piece sits. Nothing is read off the op list: an imported mesh is
measured the same way.

Tolerance. The disc is the tool's radius less TOLERANCE and a remainder thinner than TOLERANCE is
dropped, so a facet's worth of chord is not a defect and a hole the tool's own size counts as cut (a
plunge). A sharp inside corner therefore reads about 2.5 % under its textbook (1 - pi/4) r^2.

What this is not. It is reachability, not a toolpath and not a surface finish: a slope comes out as
steps one layer high. The stock is the part's bounding box plus `allowance`. The tool is a flat end
mill as wide as its shank, so the holder never collides. Workholding (tabs, tape, flip registration)
is not modelled. Tool reach, fixturing margin and Z clearance are assumptions to check on the machine.

MACHINES holds published figures, not measurements, as core/printers.py does for beds.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, fields

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType

from .critique import Issue
from .evaluate import RawMesh, Shape
from .fit import UNIT_TO_MM

__all__ = ["Tool", "Machine", "Lump", "Report", "MachiningError", "TOOLS", "MACHINES", "DEFAULT_TOOL", "SETUPS",
           "analyse", "reports", "explain", "issues"]

TOLERANCE = 0.02         # mm: a remainder thinner than this counts as cut (the Z1's stated motion resolution)
STEP = 0.5               # mm: layer height where the section changes with height (slopes, curves)
MAX_LAYERS = 240         # per setup; a taller part gets thicker layers, never more of them
MAX_LEVELS = 96          # more distinct vertex heights than this is a curved part: slice it evenly
ARC = 96                 # segments per turn when the tool's disc is swept around a corner
NEGLIGIBLE = 0.1         # mm^3: a remainder smaller than this is noise, not a defect
DEGENERATE = 1e-6        # mm^2: a contour smaller than this is a boolean's leftover loop, not a region,
THREAD = 1e-5            # mm: and so is one thinner than this on average (twice its area over its length)
NOISE = 0.05             # two sets of setups that leave within this share of each other are a tie, and the
                         # earlier wins: setups about different axes step a curve differently by about this
GAIN = 1 / 3             # when nothing clears the part, a further setup must take away this share of what is
                         # left to be counted: it costs a re-clamp


class MachiningError(ValueError):
    """The part could not be measured. Raised instead of a number that might be wrong."""


@dataclass(frozen=True)
class Tool:
    """A flat end mill, millimetres. `reach` is how far below the top of the stock its tip can cut:
    the flute length, or the stick-out if the shank is relieved."""
    id: str
    name: str
    diameter: float
    reach: float

    def __post_init__(self):
        if not (self.diameter > 0 and self.reach > 0):
            raise ValueError("tool diameter and reach must be positive")


@dataclass(frozen=True)
class Machine:
    """A 3-axis mill, millimetres. `margin` is kept free on each side of the stock for the tool to pass
    and for clamps; `z_clearance` is taken off the height for tool stick-out and a safe retract. Both
    are assumptions until measured on the machine."""
    id: str
    name: str
    vendor: str
    travel_x: float
    travel_y: float
    travel_z: float
    margin: float = 5.0
    z_clearance: float = 20.0
    tool_change: str = "manual"

    def usable(self) -> tuple[float, float, float]:
        """The largest stock: travel less the margin on both sides in the plane, less the clearance in Z."""
        return (self.travel_x - 2 * self.margin, self.travel_y - 2 * self.margin, self.travel_z - self.z_clearance)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Reach figures are typical flute lengths for these diameters, not a vendor's list: edit to match your bits.
TOOLS: dict[str, Tool] = {t.id: t for t in (
    Tool("em_6", "6 mm flat end mill", 6.0, 20.0),
    Tool("em_3175", "1/8 in flat end mill", 3.175, 12.0),
    Tool("em_2", "2 mm flat end mill", 2.0, 8.0),
    Tool("em_1", "1 mm flat end mill", 1.0, 4.0),
)}

# Makera's published work area for the Z1 (200 x 200 x 100 mm). Its tool change is a manual lever and
# 1/8 in is the standard collet, so the default tool is em_3175 and every extra tool is a stop.
MACHINES: dict[str, Machine] = {m.id: m for m in (
    Machine("makera_z1", "Z1", "Makera", 200.0, 200.0, 100.0),
)}
DEFAULT_TOOL = {"makera_z1": "em_3175"}

# A setup is named by the face that is up, in the program's frame (Y up, front = +Z). Each matrix turns
# the program frame so that face looks along +Z, the spindle axis; all six are proper rotations. A face
# and its opposite clamp about the same axis (AXES): they are measured in the first one's frame.
SETUPS: dict[str, np.ndarray] = {
    "top": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),         # +Y up
    "bottom": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),      # -Y up
    "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),       # +X up
    "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),        # -X up
    "front": np.eye(3, dtype=np.float64),                                           # +Z up
    "back": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),       # -Z up
}
AXES: tuple[tuple[str, str], ...] = (("top", "bottom"), ("right", "left"), ("front", "back"))
_AXIS_OF = {name: (i, side) for i, pair in enumerate(AXES) for side, name in enumerate(pair)}


@dataclass(frozen=True)
class Lump:
    """One connected piece of the remainder, in the program's frame, millimetres."""
    volume: float
    centre: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Report:
    """One part on one machine with one tool. Volumes are mm^3. `to_remove` is the stock less the part,
    and `pieces` how many separate bodies the part is (each would come loose as its own piece). `setups`
    is the fewest that fit the machine and leave the least; `undercut`, `corner` and `deep` split what
    those setups still leave: hidden from every one of them, too tight for the tool, below its reach."""
    part: str
    label: str
    machine: str
    tool: str
    stock: tuple[float, float, float]
    fits: bool
    setups: tuple[str, ...]
    tried: int
    volume: float
    to_remove: float
    pieces: int
    undercut: float
    corner: float
    deep: float
    lumps: tuple[Lump, ...]
    clearing_tool: str | None = None
    clearing_setups: tuple[str, ...] = ()

    @property
    def leftover(self) -> float:
        return self.undercut + self.corner + self.deep

    @property
    def machinable(self) -> bool:
        return self.fits and self.leftover <= NEGLIGIBLE

    def as_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["lumps"] = [{"volume": l.volume, "centre": l.centre, "size": l.size} for l in self.lumps]
        d["leftover"], d["machinable"] = self.leftover, self.machinable
        return d


def _turn(m: Manifold, R: np.ndarray) -> Manifold:
    return m.transform(np.hstack([R, np.zeros((3, 1))]))


def _rect(x0: float, y0: float, x1: float, y1: float) -> CrossSection:
    return CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


def _tidy(c: CrossSection) -> tuple[CrossSection, float, int]:
    """c without its degenerate contours, the length of its boundary, and how many contours it has.
    Booleans between shapes that share an edge (a tooth tip on the edge of the stock) leave loops of
    1e-10 to 1e-8 mm^2, some wound backwards, and one of those can make an offset return nothing, or
    everything. That shipped once in this module as "this gear is fully machinable"; the gears are in
    tests/test_machining.py."""
    polys = c.to_polygons()
    if not polys:
        return c, 0.0, 0
    area = np.array([0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])) for q in polys])
    edge = np.array([float(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).sum()) for q in polys])
    real = (np.abs(area) > DEGENERATE) & (2 * np.abs(area) > THREAD * edge)
    if real.all():
        return c, float(edge.sum()), len(polys)
    return (CrossSection([q for q, ok in zip(polys, real) if ok], FillRule.Positive), float(edge[real].sum()),
            int(real.sum()))


def _open(c: CrossSection, r: float, arc: int = ARC) -> CrossSection:
    """c with everything a disc of radius r cannot enter taken away. Whatever goes lies within r of the
    boundary, and that band is no larger than 2 r per length of boundary plus a disc per contour;
    shrinking never adds and growing back never takes. An offset that breaks one of those has failed
    (it can return nothing, or everything), and that is raised, never returned as a measurement."""
    if r <= 0 or c.is_empty():
        return c
    c, edge, loops = _tidy(c)
    if c.is_empty():
        return c
    shrunk = c.offset(-r, JoinType.Round, 2.0, arc)
    grown = shrunk.offset(r, JoinType.Round, 2.0, arc)
    out = grown ^ c
    whole, band = c.area(), 1.02 * (2 * edge * r + loops * math.pi * r * r) + 1e-6
    if not (whole - band <= shrunk.area() <= whole + 1e-6 and grown.area() >= shrunk.area() - 1e-6
            and out.area() >= whole - band):
        raise MachiningError(f"opening a {whole:.3f} mm^2 section by {r:g} mm gave {shrunk.area():.3f} shrunk and "
                             f"{out.area():.3f} back, outside what its {edge:.1f} mm of boundary allows: the "
                             "offset failed and nothing was measured")
    return out


def _clean(c: CrossSection, tol: float) -> CrossSection:
    """c without slivers thinner than tol: a facet's worth of remainder is cut, not a defect."""
    return _open(c, tol / 2, 8) if tol > 0 else _tidy(c)[0]


def _same(a: CrossSection, b: CrossSection) -> bool:
    if a.is_empty() or b.is_empty():
        return a.is_empty() and b.is_empty()
    return abs(a.area() - b.area()) <= 1e-9 * max(1.0, a.area()) and np.allclose(a.bounds(), b.bounds(), atol=1e-6)


def _layers(m: Manifold, z0: float, z1: float, cuts: tuple[float, ...], step: float) -> list[tuple[float, float]]:
    """Layers (bottom, top) from z0 to z1. A stretch where the section does not change is one layer
    whatever its height, so a prismatic part costs a handful of layers and is measured exactly; a
    stretch where it changes is cut every `step`. `cuts` are heights a layer must not straddle."""
    mesh = getattr(m, "to_mesh64", m.to_mesh)()                             # double precision where the kernel has it
    z = np.unique(np.round(np.asarray(mesh.vert_properties, dtype=np.float64)[:, 2], 4))
    z = z[(z > z0 + 1e-6) & (z < z1 - 1e-6)]
    extra = [c for c in cuts if z0 + 1e-6 < c < z1 - 1e-6]
    step = max(step, (z1 - z0) / MAX_LAYERS)
    if len(z) > MAX_LEVELS:
        marks = sorted({z0, z1, *extra})
        return [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n)
                for a, b in zip(marks[:-1], marks[1:]) for n in [max(1, math.ceil((b - a) / step))] for i in range(n)]
    marks = sorted({z0, z1, *z.tolist()})
    out: list[tuple[float, float]] = []
    for a, b in zip(marks[:-1], marks[1:]):
        e = min(1e-3, (b - a) / 4)
        n = 1 if _same(m.slice(a + e), m.slice(b - e)) else max(1, math.ceil((b - a) / step))
        edges = sorted({a, b, *(a + (b - a) * i / n for i in range(1, n)), *(c for c in extra if a < c < b)})
        out.extend(zip(edges[:-1], edges[1:]))
    return out


_NONE = CrossSection()


@dataclass
class _Axis:
    """Both setups about one axis, in the frame of the first (Z up). Per layer: what must go (`need`),
    and per setup what the tool can see (`free`) and whether the layer is inside its reach (`within`).
    What a setup leaves is worked out from these on demand, because seeing is cheap and fitting is not."""
    R: np.ndarray
    layers: list[tuple[float, float]]
    need: list[CrossSection]
    free: tuple[list[CrossSection], list[CrossSection]]
    within: tuple[list[bool], list[bool]]
    r: float
    tol: float
    _fits: dict = field(default_factory=dict)
    _left: dict = field(default_factory=dict)

    def left(self, side: int, which: int) -> list[CrossSection]:
        """Per layer, what the setup leaves: 0 what it cannot see, 1 plus what the tool cannot fit
        into, 2 plus what lies below its reach."""
        if (side, which) not in self._left:
            if which == 0:
                got = self.free[side]
            else:
                if side not in self._fits:
                    once: dict[int, CrossSection] = {}      # an unchanged shadow is one object: open it once
                    for f in self.free[side]:
                        if id(f) not in once:
                            once[id(f)] = _open(f, self.r)
                    self._fits[side] = [once[id(f)] for f in self.free[side]]
                got = self._fits[side] if which == 1 else \
                    [f if w else _NONE for f, w in zip(self._fits[side], self.within[side])]
            self._left[(side, which)] = [_clean(n - g, self.tol) for n, g in zip(self.need, got)]
        return self._left[(side, which)]


def _axis(part: Manifold, R: np.ndarray, lo: np.ndarray, hi: np.ndarray, tool: Tool, tol: float, step: float) -> _Axis:
    """Lay out the two setups about one axis. `part` and the stock corners are already in the frame."""
    pad = 2 * tool.diameter + 1.0
    plane = _rect(lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad)       # the tool may run off the stock
    stock = _rect(lo[0], lo[1], hi[0], hi[1])
    z0, z1 = float(lo[2]), float(hi[2])
    layers = _layers(part, z0, z1, (z1 - tool.reach, z0 + tool.reach), step)
    sections = [part.slice((a + b) / 2) for a, b in layers]
    need = [(plane - s) ^ stock for s in sections]
    free: tuple[list, list] = ([_NONE] * len(layers), [_NONE] * len(layers))
    for side, order in ((0, range(len(layers) - 1, -1, -1)), (1, range(len(layers)))):
        shadow, area, seen = _NONE, -1.0, _NONE
        for k in order:                                                   # from the face the tool enters by
            grown = shadow + sections[k]
            if abs(grown.area() - area) > 1e-9 * max(1.0, area):          # the shadow grew
                shadow = grown.simplify(tol / 4) if tol > 0 else grown
                area = shadow.area()
                seen = plane - shadow
            free[side][k] = seen
    within = ([a >= z1 - tool.reach - 1e-9 for a, _ in layers], [b <= z0 + tool.reach + 1e-9 for _, b in layers])
    return _Axis(R, layers, need, free, within, max(tool.diameter / 2 - tol, 0.0), tol)


def _fits(size: np.ndarray, machine: Machine) -> bool:
    ux, uy, uz = machine.usable()
    x, y, z = (float(v) for v in size)
    return z <= uz + 1e-9 and ((x <= ux + 1e-9 and y <= uy + 1e-9) or (y <= ux + 1e-9 and x <= uy + 1e-9))


def _by_axis(setups: tuple[str, ...]) -> dict[int, tuple[int, ...]]:
    out: dict[int, list[int]] = {}
    for s in setups:
        i, side = _AXIS_OF[s]
        out.setdefault(i, []).append(side)
    return {i: tuple(sorted(v)) for i, v in out.items()}


class _Part:
    """One solid in millimetres with its stock, and what each set of setups leaves (worked out once)."""

    def __init__(self, m: Manifold, allowance: float, tol: float, step: float):
        self.m, self.tol, self.step = m, tol, step
        b = np.asarray(m.bounding_box(), dtype=np.float64)
        self.lo, self.hi = b[:3] - allowance, b[3:] + allowance
        self._axes: dict[tuple[int, str], _Axis] = {}
        self._solids: dict[tuple, Manifold] = {}
        self._volumes: dict[tuple, float] = {}

    def stock_in(self, setup: str) -> np.ndarray:
        """The stock's size with this face up: (x, y in the plane, z along the spindle)."""
        c = np.array([SETUPS[setup] @ p for p in (self.lo, self.hi)])
        return c.max(0) - c.min(0)

    def axis(self, i: int, tool: Tool) -> _Axis:
        if (i, tool.id) not in self._axes:
            R = SETUPS[AXES[i][0]]
            c = np.array([R @ p for p in (self.lo, self.hi)])
            self._axes[(i, tool.id)] = _axis(_turn(self.m, R), R, c.min(0), c.max(0), tool, self.tol, self.step)
        return self._axes[(i, tool.id)]

    def _regions(self, i: int, sides: tuple[int, ...], tool: Tool, which: int) -> list[CrossSection]:
        """Per layer, what the setups `sides` about axis i leave between them."""
        ax = self.axis(i, tool)
        if len(sides) == 1:
            return ax.left(sides[0], which)
        return [_clean(a ^ b, self.tol) for a, b in zip(ax.left(0, which), ax.left(1, which))]

    def _flat(self, i: int, sides: tuple[int, ...], tool: Tool, which: int) -> float:
        key = (i, sides, tool.id, which)
        if key not in self._volumes:
            self._volumes[key] = float(sum(c.area() * (b - a) for c, (a, b) in
                                           zip(self._regions(i, sides, tool, which), self.axis(i, tool).layers)))
        return self._volumes[key]

    def _solid(self, i: int, sides: tuple[int, ...], tool: Tool, which: int) -> Manifold:
        key = (i, sides, tool.id, which)
        if key not in self._solids:
            ax = self.axis(i, tool)
            thin = self.tol / 2                                           # arcs swept by the tool: far fewer facets
            slabs = [Manifold.extrude(c.simplify(thin) if thin > 0 else c, b - a).translate((0.0, 0.0, a))
                     for c, (a, b) in zip(self._regions(i, sides, tool, which), ax.layers) if c.area() > 0]
            self._solids[key] = _turn(Manifold.batch_boolean(slabs, OpType.Add), ax.R.T) if slabs else Manifold()
        return self._solids[key]

    def solid(self, setups: tuple[str, ...], tool: Tool, which: int = 2) -> Manifold:
        solids = [self._solid(i, sides, tool, which) for i, sides in _by_axis(setups).items()]
        return solids[0] if len(solids) == 1 else Manifold.batch_boolean(solids, OpType.Intersect)

    def leftover(self, setups: tuple[str, ...], tool: Tool, which: int = 2) -> float:
        """mm^3 the setups leave between them. One axis is summed in the plane; across axes a set
        leaves nothing if any one axis does, and is otherwise intersected as solids."""
        groups = _by_axis(setups)
        flat = [self._flat(i, sides, tool, which) for i, sides in groups.items()]
        if len(flat) == 1 or min(flat) <= NEGLIGIBLE:
            return min(flat)
        key = (tuple(sorted(groups.items())), tool.id, which)
        if key not in self._volumes:
            self._volumes[key] = float(self.solid(setups, tool, which).volume())
        return self._volumes[key]

    def _first(self, sets: list[tuple[str, ...]], tool: Tool) -> tuple[float, tuple[str, ...]]:
        """The set that leaves least; a later one must beat an earlier one by NOISE to replace it."""
        best = (self.leftover(sets[0], tool), sets[0])
        for c in sets[1:]:
            v = self.leftover(c, tool)
            if v < best[0] * (1 - NOISE) - 1e-9:
                best = (v, c)
        return best

    def search(self, ok: list[str], tool: Tool, max_setups: int, clear_only: bool = False
               ) -> tuple[float, tuple[str, ...]] | None:
        """The fewest setups that leave nothing: sets are tried smallest first, a face and its opposite
        before a quarter turn, otherwise in the order of `ok`. What a set cannot see it cannot cut, so
        that is looked at first, and before any set across axes is intersected, every setup at once is:
        if all of them leave something, none will do. Then (unless `clear_only`, which returns None)
        the answer is the single setup that leaves least, with further setups added one at a time while
        each takes away GAIN of what is left."""
        limit = min(max_setups, len(ok))
        order = lambda c: tuple(s for s in ok if s in c)                  # noqa: E731
        sets = [c for n in range(1, limit + 1) for c in sorted(itertools.combinations(ok, n),
                                                               key=lambda c: len(_by_axis(c)))]
        floor = None                                                      # what every setup at once leaves
        for c in sets:
            if len(_by_axis(c)) > 1 and floor is None:
                floor = self.leftover(tuple(ok), tool)
                if floor > NEGLIGIBLE:
                    break
            if self.leftover(c, tool, 0) <= NEGLIGIBLE and self.leftover(c, tool) <= NEGLIGIBLE:
                return self.leftover(c, tool), c
        if clear_only:
            return None
        left, chosen = self._first([(s,) for s in ok], tool)
        while len(chosen) < limit:
            if floor is not None and floor >= left * (1 - GAIN):          # not even all of them gain enough
                break
            v, more = self._first([order((*chosen, s)) for s in ok if s not in chosen], tool)
            if v >= left * (1 - GAIN):
                break
            left, chosen = v, more
        return left, chosen


def analyse(m: Manifold, machine: Machine, tool: Tool, *, units: str = "mm", part: str = "", label: str = "",
            max_setups: int = 3, allowance: float = 0.0, tools: tuple[Tool, ...] = (),
            tol: float = TOLERANCE, step: float = STEP) -> Report:
    """Measure one closed solid. Setups are tried fewest first, as built (top) before flipped (bottom)
    before the four sides, and only with the stock lying a way the machine holds (_Part.search). When
    the tool is what holds it back, `tools` (smaller cutters) get the same search and the largest that
    clears the part is named with its setups."""
    if units not in UNIT_TO_MM:
        raise ValueError(f"units must be one of {sorted(UNIT_TO_MM)}")
    if isinstance(m, RawMesh) or m.status().name != "NoError" or m.is_empty():
        raise ValueError("machinability needs a closed solid")
    if not (isinstance(max_setups, int) and 1 <= max_setups <= 6):
        raise ValueError("max_setups must be 1..6")
    if allowance < 0 or tol < 0 or step <= 0:
        raise ValueError("allowance and tol must not be negative, and step must be positive")
    p = _Part(m.scale([UNIT_TO_MM[units]] * 3), float(allowance), float(tol), float(step))
    size = p.hi - p.lo
    volume = float(p.m.volume())
    ok = [s for s in SETUPS if _fits(p.stock_in(s), machine)]
    base = dict(part=part, label=label or part, machine=machine.id, tool=tool.id, tried=max_setups,
                stock=tuple(float(v) for v in size), volume=volume,
                pieces=sum(1 for q in p.m.decompose() if q.volume() > 0),      # a cavity's shell is not a piece
                to_remove=max(0.0, float(np.prod(size)) - volume))
    if not ok:
        return Report(**base, fits=False, setups=(), undercut=0.0, corner=0.0, deep=0.0, lumps=())

    total, setups = p.search(ok, tool, max_setups)
    if total <= NEGLIGIBLE:
        return Report(**base, fits=True, setups=setups, undercut=0.0, corner=0.0, deep=0.0, lumps=())
    hidden, tight = p.leftover(setups, tool, 0), p.leftover(setups, tool, 1)
    lumps = sorted((Lump(float(l.volume()), *_box(l)) for l in p.solid(setups, tool).decompose()),
                   key=lambda l: -l.volume)
    clearing: tuple[str | None, tuple[str, ...]] = (None, ())
    if tight - hidden > NEGLIGIBLE:
        for t in sorted((t for t in tools if t.diameter < tool.diameter), key=lambda t: -t.diameter):
            found = p.search(ok, t, max_setups, clear_only=True)
            if found is not None:
                clearing = (t.id, found[1])
                break
    return Report(**base, fits=True, setups=setups, undercut=hidden, corner=max(0.0, tight - hidden),
                  deep=max(0.0, total - tight), lumps=tuple(lumps[:8]),
                  clearing_tool=clearing[0], clearing_setups=clearing[1])


def _box(m: Manifold) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    b = np.asarray(m.bounding_box(), dtype=np.float64)
    return tuple(float(v) for v in (b[:3] + b[3:]) / 2), tuple(float(v) for v in b[3:] - b[:3])


def reports(shapes: list[Shape], units: str, machine: Machine, tool: Tool, **kw) -> list[Report]:
    """A report for every closed solid in the build, each measured alone as it stands in the build."""
    return [analyse(s.manifold, machine, tool, units=units, part=s.id, label=s.label, **kw)
            for s in shapes if not isinstance(s.manifold, RawMesh)]


def _faces(setups: tuple[str, ...]) -> str:
    return "the " + " and ".join(setups) + (" face" if len(setups) == 1 else " faces")


def _where(l: Lump) -> str:
    return (f" The largest piece is {' x '.join(f'{v:.1f}' for v in l.size)} mm, centred at "
            f"({', '.join(f'{v:.1f}' for v in l.centre)}).")


def explain(r: Report, machine: Machine, tool: Tool, tools: tuple[Tool, ...] = (), subject: str = "") -> list[Issue]:
    """One report as critic issues (core/critique.py), so it can go back to a model as an edit. `fix`
    is a defect in the part that a change of shape repairs; `warn` is a cost (a re-clamp)."""
    names = {t.id: t.name for t in tools}
    if not r.fits:
        u = machine.usable()
        return [Issue("stock_fit", "fix", subject, f"{r.label} needs a {r.stock[0]:.0f} x {r.stock[1]:.0f} x "
                      f"{r.stock[2]:.0f} mm block. The {machine.vendor} {machine.name} holds {u[0]:.0f} x {u[1]:.0f} mm "
                      f"in the plane and {u[2]:.0f} mm under the tool, and no face up fits. Make it smaller or "
                      "split it.")]
    out: list[Issue] = []
    if r.pieces > 1:
        out.append(Issue("pieces", "warn", subject, f"{r.label} is {r.pieces} separate pieces measured in one block: "
                         "each comes loose as it is cut out, so hold them with tabs or make them separate parts."))
    if len(r.setups) > 1:
        out.append(Issue("setups", "warn", subject, f"{r.label} needs {len(r.setups)} setups ({_faces(r.setups)} up "
                         "in turn): it is turned and located again between them."))
    at = _where(r.lumps[0]) if r.lumps else ""
    if r.undercut > NEGLIGIBLE:
        out.append(Issue("undercut", "fix", subject, f"{r.label}: {r.undercut:.1f} mm^3 that must be cut away is "
                         f"hidden from the tool. With {_faces(r.setups)} up it sits under material, and no set of up "
                         f"to {r.tried} setups does better. Open it to an outside face, or split the part there.{at}"))
    if r.corner > NEGLIGIBLE:
        fix = (f"the {names[r.clearing_tool]} clears it with {_faces(r.clearing_setups)} up"
               if r.clearing_tool in names else "no smaller tool in the list clears it")
        out.append(Issue("inside_corner", "fix", subject, f"{r.label}: {r.corner:.1f} mm^3 is left in inside "
                         f"corners, slots or holes tighter than the {tool.name} (radius {tool.diameter / 2:.2f} mm). "
                         f"Round inside corners to at least {math.ceil(tool.diameter * 5 + 1) / 10:.1f} mm and open "
                         f"slots and holes past {tool.diameter:g} mm; {fix}.{at}"))
    if r.deep > NEGLIGIBLE:
        out.append(Issue("too_deep", "fix", subject, f"{r.label}: {r.deep:.1f} mm^3 lies more than {tool.reach:g} mm "
                         f"below the face it is cut from, past the reach of the {tool.name}. Make it shallower, open "
                         f"it from the opposite face, or use a longer tool.{at}"))
    return out


def issues(shapes: list[Shape], units: str, machine: Machine, tool: Tool, **kw) -> list[Issue]:
    """Every part of the build as critic issues. An open surface, or a part the measurement fails on,
    is said so (`warn`) and never passed as machinable."""
    kw.setdefault("tools", tuple(TOOLS.values()))
    out: list[Issue] = []
    for s in shapes:
        if isinstance(s.manifold, RawMesh):
            out.append(Issue("not_solid", "warn", s.component, f"{s.label} is an open surface, so what a mill "
                             "would leave of it cannot be measured."))
            continue
        try:
            r = analyse(s.manifold, machine, tool, units=units, part=s.id, label=s.label, **kw)
        except MachiningError as e:
            out.append(Issue("unmeasured", "warn", s.component, f"{s.label} could not be measured for machining: {e}."))
            continue
        out.extend(explain(r, machine, tool, kw["tools"], s.component))
    return out
