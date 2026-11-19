# Replicator: rules for agents and contributors

Replicator is an engineering and research workshop for nearly anything physical: houses and floor plans, electronics,
household objects, machines, concepts. A sentence becomes a blueprint, the blueprint becomes a **BuildProgram**
(ordered solid-modelling ops), manifold3d builds it, a critic measures it, and the Qt Quick 3D workspace shows it
step by step. You can dissect it, reverse-engineer an imported model, explain it, render it and hand it to Blender.
Models are any OpenAI-compatible endpoint the user names; agents drive it over MCP. Printing is optional: every
feature that does not need a printer works without one. Read `README.md` for the product, `docs/DESIGN.md` for
direction, `docs/ROADMAP.md` for what is next and `docs/REQUIREMENTS-workbench.md` for where it is going.

## Commands

```sh
pip install -e ".[all]"                                         # once, in a venv
python -m pytest -q                                             # all tests, under 30 s; must stay green
python -m tools.check path/to/program.json                      # validate, build, mass, critic; exit 1 on a FIX issue
python -m tools.machinability path/to/program.json              # per part on a 3-axis mill: stock, setups, what stays uncut
python -m replicator.app replicator/fixtures/rocket_engine.json --software \
    --shot out/x.png [--mode build|dissect|reverse|showcase|present] [--ui materials|commander|preferences]
python -m tools.rocket_fixture                                  # regenerate the reference rocket engine
replicator [program.json | model.glb | bundle.zip]              # the real window
```

`--shot` needs a display (a Wayland or X11 session): Qt's `offscreen` platform cannot run Quick3D. Look at every shot you take; a shot is the only proof a QML change works.
`--shot` grabs a hidden window, which lays out only during the grab, so overlays projected from the 3D view (Dissect
callouts) and narrow-window layouts can come out wrong. For those, `tools/shot.sh program.json out.png [WxH] [cue ...]`
takes one frame of the real window in a headless gamescope. Shots and recordings use an empty services.json unless
`REPLICATOR_SERVICES` is set, so a frame never shows the operator's endpoints or paths.

## Where things live

| Path | Owns |
|---|---|
| `replicator/core/program.py` | BuildProgram schema and validation, JSON Patch, versions on disk, `component_map`, component delete and duplicate |
| `replicator/core/evaluate.py` | the kernel: op dispatch, per-op cache keyed by hash, build step views |
| `replicator/core/ops_advanced.py`, `ops_engineering.py` | op tables (`PRIMITIVES`, `MODIFIERS`): loft, pipe, helix, gear, deform; tube, nozzle, rotor, grille |
| `replicator/core/critique.py` | accuracy checks (`CHECKS` tuple): floating, buried, clipping, off-axis, size (warn >15%, fix >40%), missing, tips, roof (buildings: covers the walls, no V where a pitch is planned, not into the rooms, stated pitch) |
| `replicator/core/physics.py` | mass, centre of gravity, stands-or-tips, densities by material name |
| `replicator/core/materials.py` | PBR presets (`PRESETS`, grouped Aerospace, Metal, Polymer, Paint, Glow), paint and whole-build swap |
| `replicator/core/importers.py` | `ROUTES`: every importable extension and how it becomes a Y-up GLB |
| `replicator/core/reverse.py`, `assembly.py`, `explain.py` | mesh decomposition and ledger, components and explode, checked explanations |
| `replicator/core/blender_link.py`, `render.py` | Blender MCP round trip, Cycles stills (CPU) |
| `replicator/core/step.py`, `export.py` | STEP (faceted B-rep, one solid per lump, cavities as voids), STL, 3MF, GLB, OBJ |
| `replicator/core/drawing.py` | A3 sheets: third-angle views, section or floor plan (buildings), dimensions, parts list |
| `replicator/core/machining.py` | machinability on a 3-axis mill: machine and tool profiles (`MACHINES`, `TOOLS`), fewest setups, what stays uncut and why (hidden, too tight, too deep), as critic issues; `tools/machinability.py` prints it per part |
| `replicator/mcp_server.py` | the workshop as MCP tools; every write validated and built first; `.mcp.json` registers it |
| `replicator/knowledge/*.md` | curated cards (real parts and proportions), picked by keyword into the specify prompts |
| `replicator/llm/client.py`, `llm/prompts/*.md` | specify (`client.dream`: blueprint, program, repair, critique round), edit (optionally scoped), naming, explain; `engines()` = built-ins plus `endpoints` from services.json |
| `replicator/qt/bridge.py` | the `studio` QObject: every property and slot QML uses |
| `replicator/qml/Main.qml` | layout, workspace state and the `actions` registry (menus, shortcuts, right-click menus, Commander) |
| `replicator/qml/panels/`, `replicator/qml/viewport/`, `replicator/qml/Replicator/UI/` | one file per panel; the 3D view; design tokens and controls |
| `replicator/fixtures/` | example programs, each passes `tools.check` (the start screen lists them). `rocket_engine.json` is the accuracy reference; failing cases live in `tests/data/` (`turbofan_bad.json`) |

## Invariants

1. **One document.** The BuildProgram is the only source of truth. Panels, the critic, mass and exports are views of
   it. Every change is a new validated program pushed through `Studio._set_program` (one undo step).
2. **`core/`, `knowledge/`, `research/` import no Qt.** They are tested headless. Values handed to QML are plain
   Python types: a numpy `float64` crosses the boundary as garbage (this shipped once as "MASS -1000 g").
3. **The model chooses, the code computes.** Proportions come from ops and formulas (`nozzle` draws a Rao bell from a
   throat radius and expansion ratio), facts from knowledge cards, numbers from measurement. Never ask a model for a
   number code can work out.
4. **Measured, not guessed.** Critic issues and mass are measurements; label anything assumed (a fallback density is
   flagged `*` in the status bar). Reverse ledger kinds: observed, inferred, unknown. Explain kinds: cited, measured,
   concept.
5. **Agent edits are patches.** `client.edit` returns RFC 6902 operations; with `scope` (op ids) any change outside the
   scope is sent back to the model as an error. A job result carries `base`; if the program changed while the model
   worked, the result is dropped, never merged over the operator's work.
6. **GPU.** Never load a model in-process. Models are reached only through the engines in `client.engines()`
   (`local` plus the user's `endpoints`). Blender, slicers and converters run with `CUDA_VISIBLE_DEVICES=`. The
   window follows `replicator.gpu` (`REPLICATOR_GPU`). One model job at a time.
7. **Only what the user named.** No engine, service or network call the user did not configure. Research
   (Wikipedia, optional SearXNG) is off until the user enables it and sends only search words.
8. **Imports are untrusted.** Archives unpack into their own new folder with path, size and count checks
   (`importers.unpack`). Never execute anything from an imported file.
9. **Printing is optional.** The printer gate applies only when an active printer is set; with none, every export and
   feature works. Never make a non-print feature depend on printer state.

## BuildProgram in brief

```json
{"version": 1, "name": "...", "units": "mm",
 "materials": {"titanium_ti6al4v": {"base_color": [0.56, 0.54, 0.51], "metalness": 1.0, "roughness": 0.3}},
 "components": [{"id": "chamber", "label": "Thrust Chamber"}],
 "ops": [{"id": "chamber", "op": "nozzle", "label": "Thrust Chamber and Bell", "component": "chamber",
          "material": "titanium_ti6al4v", "params": {"throat_radius": 100, "expansion_ratio": 16, "translate": [0, 800, 0]}}]}
```

- Y up, millimetres, right-handed. Primitives are centred; axisymmetric ops (`cylinder`, `tube`, `torus`, `nozzle`,
  `rotor`, `revolve`) spin around Y. On a body lying along Z give them `"rotate": [90, 0, 0]`; the critic flags a
  ring on the centre line that spins around the wrong axis.
- Placement (`translate`, `rotate` in degrees X then Y then Z, `scale`) is applied last. Radial `array` copies rotate
  around the world origin, so place the source off-axis first.
- An op consumed by a later op disappears; only leaves are visible. A body that houses parts must be hollow (`tube`,
  `shell`, a revolve with an inner wall) or the critic reports the parts inside as buried.
- `component` groups ops into parts the UI, critic, mass, explode and Blender collections all use.
- Op parameter lists come from the tables: `client.op_reference()` prints them.

## Extending it (each is one table entry plus a test)

- **An op:** a builder in `ops_engineering.py` returning a `Manifold`, an entry in its `PRIMITIVES`
  (`name: (required, optional, _wrap_builder(fn))`), a paragraph in `llm/prompts/program.md`, a `bridge.addPrimitive`
  default and a `Create` action in `Main.qml`. The schema and validation pick it up automatically.
- **A critic check:** a function `(Build) -> list[Issue]` added to `CHECKS`. `fix` only for defects a model can repair.
- **A knowledge card:** `replicator/knowledge/<topic>.md` with `title:` and `keywords:` front matter, under 1k tokens,
  real numbers only, and a pick assertion in `tests/test_accuracy.py`.
- **A material:** a `PRESETS` row whose id contains a substance word `physics.DENSITY` knows (`titanium_...`,
  `steel_...`), so its mass is computed, not assumed.
- **An import format:** a `ROUTES` row; reuse an existing kind (`mesh`, `cad`, `blender`, `archive`).
- **A command:** one object in `actions` (`id`, `group`, `title`, `keys`, `icon`, `enabled`, `run`); menus, right-click
  menus (`partMenu`, `sceneMenu`) and Commander list ids. Never hard-code a colour in a panel: use `Theme`.
- **A bridge slot:** validate on a copy, build it with `self._ev.step(prog)` (shared cache), then `_set_program`.
  Model work goes through `_run_job` / `_agent_edit`; never touch `self._program` or Qt objects from the job thread.

## Driving it as an agent

- Write or patch a program, run `tools.check`, fix every `FIX`, then take a `--shot` in build and showcase and look.
- To change one part, scope the edit to that component's ops (`program.component_ops`), as the Rework action does.
- Blender: *Open in Blender* puts one collection per component, objects named by label with custom properties
  `replicator_id` and `replicator_component`; the vendored mcp-for-blender socket listens on `localhost:9876` only.
  *Pull back* reverse-engineers the live scene into Reverse.
- Known limits: no fluid or structural simulation (strength and CFD are planned, not built); STEP/IGES is tested on
  one SolidWorks part (`tests/data/featuretype.STEP`), not yet on large assemblies; `round` erodes then regrows, so
  features thinner than twice its radius vanish (with an error).

## Git

Small commits, each one tested. No AI attribution lines in commits or PRs.
