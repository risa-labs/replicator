"""Machinability (core/machining.py): what a 3-axis mill leaves of a part, on the Makera Z1 (usable stock
190 x 190 x 80 mm) with the 1/8 in end mill (radius 1.5875 mm, reach 12 mm).

Every expected volume is worked out by hand from the op parameters. A cylinder is a 64-gon here, so its
section is 32 r^2 sin(2 pi / 64). A sharp inside corner keeps (1 - pi/4) rho^2 per millimetre of depth,
where rho is the tool radius less the 0.02 mm tolerance (1.5675), because that is the disc the module
sweeps. The tests at the end check it against a method that shares none of its code: the section
drawn into pixels and opened with distance transforms."""
import math
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from bench.tasks import flange
from replicator.core import machining as M
from replicator.core.evaluate import Evaluator, RawMesh, Shape
from replicator.core.program import check

Z1 = M.MACHINES["makera_z1"]
EM = M.TOOLS["em_3175"]
ALL = tuple(M.TOOLS.values())
RHO = EM.diameter / 2 - M.TOLERANCE
GON = 32 * math.sin(2 * math.pi / 64)          # area of the unit 64-gon, where a circle has pi
CUT = {"id": "part", "op": "boolean", "label": "Part", "inputs": ["b", "p"], "params": {"mode": "difference"}}


def _shapes(ops, units="mm"):
    return Evaluator().step(check({"version": 1, "name": "t", "units": units, "ops": ops}))


def _report(ops, units="mm", tool=EM, **kw):
    (r,) = M.reports(_shapes(ops, units), units, Z1, tool, tools=ALL, **kw)
    return r


def _flange(p: dict) -> list[dict]:
    """The exact solid the benchmark's flange prompt describes (the same ops as tests/test_bench.py)."""
    return [
        {"id": "disc", "op": "cylinder", "params": {"radius": p["od"] / 2, "height": p["t"]}},
        {"id": "bore_cut", "op": "cylinder", "params": {"radius": p["bore"] / 2, "height": p["t"] + 4}},
        {"id": "bolt", "op": "cylinder", "params": {"radius": p["h"] / 2, "height": p["t"] + 4, "translate": [p["pcd"] / 2, 0, 0]}},
        {"id": "bolts", "op": "array", "input": "bolt", "params": {"mode": "radial", "count": p["n"], "angle": 360, "axis": "y"}},
        {"id": "flange", "op": "boolean", "label": "Flange", "inputs": ["disc", "bore_cut", "bolts"], "params": {"mode": "difference"}},
    ]


def _block(size, *cuts):
    ids = [c["id"] for c in cuts]
    return [{"id": "b", "op": "box", "params": {"size": list(size)}}, *cuts, dict(CUT, inputs=["b", *ids])]


# --- the benchmark flange ---------------------------------------------------------------------------------

def test_the_benchmark_flange_is_machinable_and_needs_one_flip():
    p = flange.instance(13).params                      # od 110, 20 thick, bore 40, eight 8 mm holes
    shapes = _shapes(_flange(p))
    (r,) = M.reports(shapes, "mm", Z1, EM, tools=ALL)
    assert r.fits and r.machinable and r.leftover == 0.0
    assert r.setups == ("top", "bottom")                # 20 mm through, and the tool reaches 12 from each face
    assert r.stock == pytest.approx((110.0, 20.0, 110.0), abs=0.1)
    assert [(i.kind, i.severity) for i in M.issues(shapes, "mm", Z1, EM)] == [("setups", "warn")]


def test_a_tool_that_reaches_through_does_the_flange_in_one_setup():
    r = _report(_flange(flange.instance(13).params), tool=M.TOOLS["em_6"])      # reach 20, holes 8 mm
    assert r.machinable and r.setups == ("top",)


@pytest.mark.parametrize("seed", range(1, 13))
def test_every_flange_instance_is_machinable_and_flips_only_when_thicker_than_the_reach(seed):
    p = flange.instance(seed).params                    # holes 8 to 12 mm, bore 20 mm up: all wider than the tool
    r = _report(_flange(p))
    assert r.machinable
    assert r.setups == (("top",) if p["t"] <= EM.reach else ("top", "bottom"))


# --- what the tool cannot fit into ---------------------------------------------------------------------------

def test_a_square_pocket_keeps_its_four_corners():
    # 30 x 30 pocket, 10 deep, open at the top of a 60 x 20 x 60 plate
    ops = _block((60, 20, 60), {"id": "p", "op": "box", "params": {"size": [30, 10, 30], "translate": [0, 5, 0]}})
    r = _report(ops)
    assert r.setups == ("top",)
    assert (r.undercut, r.deep) == (0.0, 0.0)
    assert r.corner == pytest.approx(4 * (1 - math.pi / 4) * RHO ** 2 * 10, rel=0.01)       # 21.09
    assert len(r.lumps) == 4 and all(l.size[1] == pytest.approx(10.0, abs=0.01) for l in r.lumps)
    assert r.clearing_tool is None                      # a sharp corner beats every round tool
    (i,) = M.issues(_shapes(ops), "mm", Z1, EM)
    assert (i.kind, i.severity) == ("inside_corner", "fix")
    assert "at least 1.7 mm" in i.message and "no smaller tool" in i.message


def test_the_same_pocket_with_two_millimetre_corners_is_clean():
    posts = [{"id": f"c{k}", "op": "cylinder", "params": {"radius": 2, "height": 10, "translate": [sx * 13, 5, sz * 13]}}
             for k, (sx, sz) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])]
    ops = [{"id": "b", "op": "box", "params": {"size": [60, 20, 60]}}, *posts,
           {"id": "p", "op": "hull", "inputs": [c["id"] for c in posts]}, CUT]
    r = _report(ops)
    assert r.machinable and r.setups == ("top",)


def test_a_hole_narrower_than_the_tool_stays_and_the_tool_that_clears_it_is_named():
    # a 2 mm hole through a 10 mm plate: nothing of it is cut. The 2 mm mill reaches 8, so it needs both faces
    ops = _block((40, 10, 40), {"id": "p", "op": "cylinder", "params": {"radius": 1, "height": 14}})
    r = _report(ops)
    assert r.corner == pytest.approx(GON * 1 ** 2 * 10, rel=1e-3)                           # 31.37
    assert (r.clearing_tool, r.clearing_setups) == ("em_2", ("top", "bottom"))
    (i,) = M.issues(_shapes(ops), "mm", Z1, EM)
    assert "the 2 mm flat end mill clears it with the top and bottom faces up" in i.message


# --- what the tool cannot see, and which face is which -------------------------------------------------------

@pytest.mark.parametrize("face, rotate, translate", [
    ("top", [0, 0, 0], [0, 11, 0]), ("bottom", [0, 0, 0], [0, -11, 0]),
    ("right", [0, 0, 90], [11, 0, 0]), ("left", [0, 0, 90], [-11, 0, 0]),
    ("front", [90, 0, 0], [0, 0, 11]), ("back", [90, 0, 0], [0, 0, -11]),
])
def test_a_blind_hole_is_cut_from_the_face_it_opens_on(face, rotate, translate):
    # a 30 mm cube with one 8 mm hole, 9 deep, drilled into one face: only that face up reaches it
    hole = {"id": "p", "op": "cylinder", "params": {"radius": 4, "height": 10, "rotate": rotate, "translate": translate}}
    r = _report(_block((30, 30, 30), hole))
    assert r.machinable and r.setups == (face,)


def test_a_cross_hole_takes_the_side_faces_and_both_when_longer_than_the_reach():
    hole = {"id": "p", "op": "cylinder", "params": {"radius": 4, "height": 44, "rotate": [0, 0, 90]}}      # along X
    assert _report(_block((10, 20, 40), hole)).setups == ("right",)
    r = _report(_block((20, 20, 40), hole))
    assert r.machinable and r.setups == ("right", "left")
    r = _report(_block((40, 20, 40), hole))             # 12 mm in from each side leaves 16 mm in the middle
    assert r.setups == ("right", "left")
    assert (r.undercut, r.corner) == (0.0, 0.0)
    assert r.deep == pytest.approx(GON * 4 ** 2 * 16, rel=1e-3)                             # 802.96
    assert r.lumps[0].size == pytest.approx((16.0, 8.0, 8.0), abs=0.02)
    assert r.lumps[0].centre == pytest.approx((0.0, 0.0, 0.0), abs=0.02)


def test_a_blind_pocket_deeper_than_the_reach_keeps_its_bottom():
    # 20 mm bore, 30 deep, in a 40 mm cube: the tool stops at 12, and no other face opens on it
    ops = _block((40, 40, 40), {"id": "p", "op": "cylinder", "params": {"radius": 10, "height": 30, "translate": [0, 5, 0]}})
    r = _report(ops)
    assert r.setups == ("top",)
    assert r.deep == pytest.approx(GON * 10 ** 2 * 18, rel=1e-3)                            # 5645.8
    assert [i.kind for i in M.issues(_shapes(ops), "mm", Z1, EM)] == ["too_deep"]


def test_a_sealed_cavity_is_hidden_from_every_face():
    ops = _block((30, 30, 30), {"id": "p", "op": "box", "params": {"size": [10, 10, 10]}})
    r = _report(ops)
    assert r.undercut == pytest.approx(1000.0, rel=1e-5) and (r.corner, r.deep) == (0.0, 0.0)
    assert r.setups == ("top",)                         # no further setup earns its re-clamp
    (i,) = M.issues(_shapes(ops), "mm", Z1, EM)
    assert (i.kind, i.severity) == ("undercut", "fix") and "no set of up to 3 setups does better" in i.message


# --- the stock on the machine -----------------------------------------------------------------------------

def test_stock_larger_than_the_travel_is_refused():
    shapes = _shapes([{"id": "part", "op": "box", "label": "Plate", "params": {"size": [250, 10, 100]}}])
    (r,) = M.reports(shapes, "mm", Z1, EM)
    assert not r.fits and not r.machinable and r.setups == ()
    (i,) = M.issues(shapes, "mm", Z1, EM)
    assert (i.kind, i.severity) == ("stock_fit", "fix")
    assert "250 x 10 x 100 mm block" in i.message and "190 x 190 mm" in i.message and "80 mm under the tool" in i.message


def test_a_tall_part_is_laid_down_to_fit_under_the_tool():
    # 70 x 150 x 70 stands 150 tall (over the 80 mm under the tool), so only its side faces can be up
    r = _report([{"id": "part", "op": "box", "params": {"size": [70, 150, 70]}}])
    assert r.fits and r.machinable and r.setups == ("right",)


def test_centimetres_measure_the_same_as_millimetres():
    mm = _report(_block((60, 20, 60), {"id": "p", "op": "box", "params": {"size": [30, 10, 30], "translate": [0, 5, 0]}}))
    cm = _report(_block((6, 2, 6), {"id": "p", "op": "box", "params": {"size": [3, 1, 3], "translate": [0, .5, 0]}}), units="cm")
    assert cm.stock == pytest.approx(mm.stock) and cm.corner == pytest.approx(mm.corner, rel=1e-6)


def test_allowance_on_the_stock_has_to_be_faced_off_from_both_sides():
    r = _report([{"id": "part", "op": "box", "params": {"size": [40, 10, 40]}}], allowance=1.0)
    assert r.stock == pytest.approx((42.0, 12.0, 42.0))
    assert r.machinable and r.setups == ("top", "bottom")
    assert r.to_remove == pytest.approx(42 * 12 * 42 - 40 * 10 * 40)


def test_a_part_in_several_pieces_is_said_to_be():
    # four posts built as one part: each is cut out cleanly, and each comes loose
    post = {"id": "post", "op": "cylinder", "params": {"radius": 3, "height": 6, "translate": [20, 0, 15]}}
    posts = {"id": "part", "op": "array", "label": "Posts", "input": "post", "params": {"mode": "radial", "count": 4, "axis": "y"}}
    shapes = _shapes([post, posts])
    (r,) = M.reports(shapes, "mm", Z1, EM)
    assert r.machinable and r.pieces == 4
    assert [(i.kind, i.severity) for i in M.issues(shapes, "mm", Z1, EM)] == [("pieces", "warn")]


# --- what is not measured is said so ------------------------------------------------------------------------

def test_an_open_surface_is_reported_and_never_passed():
    tri = RawMesh(np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float), np.array([[0, 1, 2]]))
    shape = Shape("sheet", "Sheet", "sheet", None, tri)
    with pytest.raises(ValueError, match="closed solid"):
        M.analyse(tri, Z1, EM)
    assert M.reports([shape], "mm", Z1, EM) == []
    (i,) = M.issues([shape], "mm", Z1, EM)
    assert (i.kind, i.severity) == ("not_solid", "warn")


def test_a_failed_offset_is_raised_not_returned_as_nothing_left(monkeypatch):
    # make the boundary look 1000 times shorter than it is: the opening then takes "too much" for it
    real = M._tidy
    monkeypatch.setattr(M, "_tidy", lambda c: (real(c)[0], real(c)[1] / 1000, 0))
    ops = _block((60, 20, 60), {"id": "p", "op": "box", "params": {"size": [30, 10, 30], "translate": [0, 5, 0]}})
    with pytest.raises(M.MachiningError, match="the offset failed"):
        _report(ops)
    (i,) = M.issues(_shapes(ops), "mm", Z1, EM)
    assert (i.kind, i.severity) == ("unmeasured", "warn")


def test_the_same_part_measures_the_same_twice():
    ops = _block((40, 20, 40), {"id": "p", "op": "cylinder", "params": {"radius": 4, "height": 44, "rotate": [0, 0, 90]}})
    assert _report(ops).as_dict() == _report(ops).as_dict()


# --- against an independent method: pixels and distance transforms ---------------------------------------------

def _raster_left(m, r: float, px: float) -> tuple[float, float]:
    """(mm^2 of the mid-height section a disc of radius r cannot clear from above, the part's height).
    The section is filled into pixels contour by contour, even-odd, and opened by two distance transforms."""
    rot = M._turn(m, M.SETUPS["top"])
    b = rot.bounding_box()
    pad = 2 * r + 2
    w, h = (int(math.ceil((b[i + 3] - b[i] + 2 * pad) / px)) for i in (0, 1))
    solid = np.zeros((h, w), dtype=bool)
    for q in rot.slice((b[2] + b[5]) / 2).to_polygons():
        img = Image.new("1", (w, h), 0)
        ImageDraw.Draw(img).polygon([((x - b[0] + pad) / px, (y - b[1] + pad) / px) for x, y in q], fill=1)
        solid ^= np.array(img, dtype=bool)
    free = ~solid
    eroded = ndimage.distance_transform_edt(free) >= r / px
    opened = ndimage.distance_transform_edt(~eroded) <= r / px
    lo = int(round(pad / px))
    inside = np.zeros_like(free)
    inside[lo:int(round((b[4] - b[1] + pad) / px)), lo:int(round((b[3] - b[0] + pad) / px))] = True
    return float((free & ~opened & inside).sum()) * px * px, b[5] - b[2]


@pytest.mark.parametrize("teeth, module, bore", [(20, 2, 6), (12, 4, 8)])
def test_gear_roots_are_left_and_agree_with_pixels(teeth, module, bore):
    """The regression for the worst bug this module had: both gears came back "nothing left", because a
    zero-area contour made the sliver clean-up return an empty region. A 1/8 in mill does not fit between
    module 2 teeth, and the root of every tooth space is a sharp corner whatever the module. The pixel
    figure runs high in proportion to the pixel, so it is taken at two sizes and carried to zero."""
    (s,) = _shapes([{"id": "part", "op": "gear", "label": "Gear",
                     "params": {"teeth": teeth, "module": module, "thickness": 8, "bore": bore}}])
    r = M.analyse(s.manifold, Z1, EM)
    (coarse, height), (fine, _) = _raster_left(s.manifold, RHO, 0.05), _raster_left(s.manifold, RHO, 0.025)
    assert r.setups == ("top",) and r.undercut == 0.0 and r.deep == 0.0
    assert r.corner > 50.0
    assert r.corner == pytest.approx((2 * fine - coarse) * height, rel=0.04)


@pytest.mark.parametrize("teeth, module, bore, left", [(20, 2, 6, 516.9), (12, 4, 8, 88.8)])
def test_without_the_contour_tidy_the_gears_raise_or_measure_right_but_never_pass(monkeypatch, teeth, module, bore, left):
    """Second line of defence. With the degenerate contours left in, the offset misbehaves on both gears
    (one comes back empty, one inside out); the checks in _open must turn that into an error. If a later
    manifold3d offsets them correctly, the right number is fine too. "Nothing left" never is."""
    real = M._tidy
    monkeypatch.setattr(M, "_tidy", lambda c: (c, *real(c)[1:]))
    (s,) = _shapes([{"id": "part", "op": "gear", "params": {"teeth": teeth, "module": module, "thickness": 8, "bore": bore}}])
    try:
        r = M.analyse(s.manifold, Z1, EM)
    except M.MachiningError:
        return
    assert r.corner == pytest.approx(left, rel=0.02)


def test_random_plates_agree_with_pixels():
    rng = random.Random(7)
    for _ in range(4):
        w, d, t = rng.uniform(30, 45), rng.uniform(30, 45), rng.choice([4, 6, 10])
        cuts = []
        for k in range(rng.randint(2, 5)):
            at = [rng.uniform(-w / 2 + 6, w / 2 - 6), 0, rng.uniform(-d / 2 + 6, d / 2 - 6)]
            if rng.random() < 0.4:      # widths stay clear of the tool's 3.135 mm: a pixel cannot settle a near tie
                cuts.append({"id": f"c{k}", "op": "cylinder", "params": {
                    "radius": rng.choice([rng.uniform(0.4, 1.3), rng.uniform(1.9, 5)]), "height": t + 4, "translate": at}})
            else:
                cuts.append({"id": f"c{k}", "op": "box", "params": {
                    "size": [rng.uniform(6, 14), t + 4, rng.choice([rng.uniform(0.6, 2.6), rng.uniform(3.7, 8)])],
                    "translate": at, "rotate": [0, rng.uniform(0, 180), 0]}})
        (s,) = _shapes(_block((w, t, d), *cuts))
        p = M._Part(s.manifold, 0.0, M.TOLERANCE, M.STEP)
        got = p.leftover(("top", "bottom"), EM, 1)          # both faces, so only the tool's radius is in play
        area, height = _raster_left(s.manifold, RHO, 0.04)
        edge = M._tidy(M._turn(s.manifold, M.SETUPS["top"]).slice(0.0))[1]
        assert got == pytest.approx(area * height, rel=0.05, abs=0.05 * 0.04 * edge * height + 0.5)


def test_degenerate_contours_do_not_empty_a_region():
    """A region with a zero-area loop on top of it (what a boolean leaves on a shared edge) keeps its area
    through the clean-up; without _tidy the offset can return nothing."""
    from manifold3d import CrossSection, FillRule
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    loop = [[2, 12], [2 + 1e-5, 12 + 1e-5], [2, 12 + 2e-5]]              # 1e-10 mm^2, wound backwards, topmost
    c = CrossSection([square, loop], FillRule.NonZero)
    tidy, edge, loops = M._tidy(c)
    assert tidy.num_contour() == 1 and edge == pytest.approx(40.0) and loops == 1
    assert M._clean(c, M.TOLERANCE).area() == pytest.approx(100.0, abs=1e-3)      # less four rounded corners
