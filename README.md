# CAD-DNA

A benchmark + surrogate for **CAD-level DNA origami routability**.

Existing tools (caDNAno, MagicDNA, ATHENA, DAEDALUS, oxDNA) follow
`design → routing → sequence → simulation`, so you only find out whether a
design is good *after* routing. The goal here is to insert a learned surrogate
before routing:

```
CAD geometry → heterogeneous graph → GNN → risk prediction → route or not
```

## Current status — 2026-08-27

- Target environment: Python 3.13 with PyTorch, NumPy, and NetworkX.
  `torch_geometric` is optional and only needed by `HeteroGraph.to_pyg()`.
- Completed: the whole `CAD → precheck → routing → label → hetero graph`
  pipeline. Designs, labels and graphs are all on disk.
- Dataset snapshot: `data/designs_v0` holds 1,000 designs + `index.csv` +
  `labels.csv`; `data/designs_v0_graphs` holds the matching 1,000 hetero graphs
  (99k nodes, 388k directed edges) + `graphs_index.csv`.
- Labels: 245/1000 routable; Hamiltonian cycle 464 True / 506 False / 30 unknown
  (budget); Hamiltonian path 715 / 241 / 44. 656 designs reach the search, and
  162 of those are decided *by* the search rather than by the precheck.
- Not yet implemented: the surrogate itself (`models/`) and its training split.
- Next milestone: a baseline GNN over `data/designs_v0_graphs`, scored against
  the precheck rules rather than against the majority class.

## Pipeline

```
CAD params                          cadna/params.py      [done]
   -> Feature / Bundle / Cylinder   cadna/generator.py   [done]
   -> cylinder adjacency            cadna/adjacency.py   [done]
   -> candidate crossovers          cadna/adjacency.py   [done]
   -> mate constraints              cadna/generator.py   [done]
   -> port graph                    cadna/linkgraph.py   [done]
   -> routing precheck              cadna/precheck.py    [done]
   -> scaffold routing (DFS)        cadna/routing.py     [done]
   -> staple + export gates         cadna/routing.py     [done]
   -> label record                  labels.csv           [done]
   -> graph extraction              cadna/graph.py       [done]
   -> graph + label on disk         dataset              [done]
   -> surrogate model               models/              [next]
```

## Prediction targets

| target | column | type | source |
|---|---|---|---|
| routing feasibility | `routable` | binary | the whole precheck → route → staple → export chain succeeded |
| Hamilton-cycle availability | `hamilton` | binary, `None` on timeout | a Hamiltonian cycle exists over the port graph |
| …allowing a free scaffold closure | `hamilton_path` | binary, `None` on timeout | a Hamiltonian *path* exists (see below) |
| failure diagnosis | `failure_class` | multi-class | `pairability` / `hamilton` / `geometry` / `timeout` / `staple_routing` / `export` / `scaffold_length` |
| search cost | `nodes_expanded`, `backtracks`, `elapsed_s`, `timeout` | regression | the DFS budget counters |

`scaffold_length` is not one of the original six classes. It is kept separate
because it is frequent and physically distinct: the topology routes fine, there
is simply not enough scaffold. `staple_ok` / `export_ok` / `scaffold_ok` are
also kept as their own columns, so a design that fails several gates at once
does not lose that information to the single primary `failure_class`.

## Layout

```
cadna/
  lattice.py     honeycomb (21 bp / 2 turns) and square (32 bp / 3 turns) geometry
                 + the crossover phase rule
  model.py       Feature / Bundle / Cylinder / Adjacency / CandidateCrossover /
                 Mate / Design dataclasses -- the schema of the hetero graph
  params.py      the CAD parameter space and its sampler
  generator.py   CADParams -> Design, four shape families
  adjacency.py   cylinder contacts and candidate-crossover enumeration
  io.py          JSON / JSON.gz (de)serialisation
  linkgraph.py   the port graph the router searches, and the parity invariant
  precheck.py    cheap exact obstructions + the structural feature vector
  routing.py     the DFS, the staple and export gates, and the label record
  graph.py       Design (+ label) -> hetero graph: numpy arrays, .npz, PyG
scripts/
  gen_designs.py    sample N designs into a directory + index.csv
  route_designs.py  label a directory of designs -> labels.csv
  build_graphs.py   designs + labels.csv -> one .npz per graph + graphs_index.csv
tests/
  test_generator.py
  test_routing.py
  test_graph.py
```

## Usage

```bash
pip install -r requirements.txt
for t in tests/test_*.py; do python "$t"; done   # no pytest needed
python scripts/gen_designs.py   --n 1000 --out data/designs_v0
python scripts/route_designs.py --designs data/designs_v0
python scripts/build_graphs.py  --designs data/designs_v0   # -> data/designs_v0_graphs
```

```python
from cadna import CADParams, sample_params, generate, evaluate, save_design

d = generate(sample_params(seed=42))                    # random point of the space
d = generate(CADParams(shape="polyhedron", polyhedron="cube",
                       lattice="honeycomb", helices_per_edge=2, edge_bp=63))
print(d.summary())
save_design(d, "data/cube.json.gz")

lab = evaluate(d)
print(lab.routable, lab.hamilton, lab.failure_class, lab.nodes_expanded)

from cadna import build_graph
g = build_graph(d, lab)                 # numpy hetero graph + target vector
print(g.summary())
g.save("data/cube.npz")
data = g.to_pyg()                       # torch_geometric HeteroData, if installed
```

## The routing model

The scaffold enters a helix at one end and leaves at the other, so every
cylinder has two **ports**, `lo` and `hi`, and traversing one always flips
lo ↔ hi. Consecutive cylinders are joined by a **link**:

- a **crossover** joins two helices of the same bundle. They share a bp frame,
  so a crossover near the low end of one is near the low end of the other:
  crossovers join *equal* ports.
- a **mate** joins helix ends of different bundles at a CAD vertex. Geometry
  decides the ports, so mates can join lo–hi.

A scaffold route is a Hamiltonian cycle alternating traversals and links.

**The parity invariant.** An equal-port link flips the entry port; a lo–hi link
does not; closing the cycle requires the entry port to come back to where it
started. So a valid route uses an even number of equal-port links — and a design
whose links are all crossovers (any single-bundle brick or plate) has no route
at all when its helix count is odd. That is the classic "odd number of helices"
obstruction, and it falls out of the model rather than being special-cased.
`tests/test_routing.py` checks the DFS against it directly.

**Cycle vs path.** A single row of helices is a lattice *path*, so it has no
Hamiltonian cycle — yet real flat sheets route fine, because a circular scaffold
closes itself over a long distance without needing a crossover (this is how a
Rothemund rectangle closes). `hamilton` is the strict cycle question;
`hamilton_path` allows that one free closure, implemented by adding a virtual
cylinder whose ports reach every real port. Both are reported, so the modelling
choice stays open.

## Graph schema

`cadna/graph.py` turns a `Design` (plus its label) into a `HeteroGraph`: plain
numpy arrays with the names of their columns, so nothing downstream depends on
a graph library. `to_pyg()` converts to `torch_geometric.data.HeteroData`, and
`save()` / `load()` write one `.npz` per design.

| node type | count | dim | carries |
|---|---|---|---|
| `feature` | CAD primitives | 17 | kind, length, centre + direction, cross-section params |
| `bundle` | helix groups | 17 | lattice, length in bp and turns, cross-section extent, axis |
| `cylinder` | helices | 29 | bp interval, lattice site, geometry, **port degrees**, link counts |
| `crossover` | candidate sites | 20 | bp index and phase, distance to each end, **the two ports**, `is_flip` |

| relation | edges | attr dim |
|---|---|---|
| `cylinder --adjacent-> cylinder` | 2 per adjacency (both ways) | 11 |
| `cylinder --mate-> cylinder` | 2 per mate (both ways) | 6 |
| `crossover --on-> cylinder`, `cylinder --hosts-> crossover` | 2 per crossover, each way | 4 |
| `bundle/feature --contains->` + reverses | the CAD hierarchy | — |

A bare `adjacent` edge says two helices touch but not *which ends* a scaffold
could join, so the parity invariant would be invisible. The port lives on the
crossover node (`port_a_is_lo`, `is_flip`) and on the incidence edge, which is
what `tests/test_graph.py::test_ports_survive_the_export` pins down.

**Two graph-level blocks, deliberately separate.** `graph_x` (24) is pure
CAD counting — sizes, densities, the bp budget. `precheck_x` (10) is the exact
obstruction detectors: dead ports, components, articulation points, bridges,
minimum degree, and the flip/no-flip parity. `precheck_x` alone decides every
fatal precheck class, so feeding it to a model makes those labels free; it ships
as a separate block precisely so `include_precheck=False` is a one-line
ablation and the precheck stays a baseline instead of an input.

**Targets** (`HeteroGraph.y`) are floats, with `NaN` for *unknown*: `hamilton`
is NaN on the 30 designs whose cycle search hit the budget. `timeout` flags the
search-cost targets as right-censored — `nodes_expanded` is a lower bound there,
not the cost. `failure_class_id` indexes `FAILURE_CLASSES`, and is `-1` for a
design that did not fail.

## Shape families

| shape | bundles | mates | regime |
|---|---|---|---|
| `brick` | 1 | none | multi-layer block, crossover rich |
| `plate` | 1 | none | 1–3 layer sheet, sparse and long |
| `polygon_ring` | N | vertex | planar wireframe |
| `polyhedron` | 6–12 | vertex | 3D wireframe (tetra / octa / cube), the DAEDALUS/ATHENA regime |

The sampler deliberately emits degenerate designs — odd helix counts per edge,
punched-out cross-sections, scaffolds far too short — because those are the
negative labels the surrogate has to learn.

## Crossover rule

A crossover occupies the **same bp index on both helices**, so the phase offset
must be symmetric in the two lattice sites. `lattice.py` derives it from the
direction of the inter-helix vector:

- honeycomb: 3 neighbour axes 120° apart, `step = 21` → offsets `{0, 7, 14}`
- square: 2 axes 90° apart with parity-alternating helix phase, `step = 32`
  → offsets `{0, 8, 16, 24}`

The tables are isolated in `Lattice.crossover_offset` so they can be swapped for
the exact caDNAno lookup tables without touching anything downstream.

## Cross-section repair

In the honeycomb lattice the third neighbour alternates up/down with parity, so
a plain rectangular `(row, col)` block grows **dangling corner helices** with a
single neighbour — which no scaffold cycle can pass through. Left alone this
made ~35% of bricks and plates trivially unroutable for an uninteresting reason.
The generator therefore drops dangling helices and keeps the largest connected
patch, exactly as a CAD tool would (a 3×3 honeycomb rectangle becomes a clean
6-helix ring). The repair is skipped when it would destroy the design rather
than tidy it — a single row of helices is a lattice path and would be eaten from
both ends — and `repair_cross_section=False` keeps the dangling-helix class in
the dataset as a deliberate minority.

## Baselines to beat

Every hard precheck in `precheck.py` is an *exact* obstruction — dead port,
disconnected, degree < 2, cut vertex, parity — so a surrogate that only
reproduces them has learned nothing. The interesting designs are the ones that
pass the precheck and are then decided by search, or that time out.

## Known simplifications

- Inter-bundle contacts are detected and stored as graph edges, but crossovers
  are only enumerated for them when `params.inter_bundle_crossovers` is set,
  since two bundles share no bp frame. Wireframe bundles meet at angles, so this
  is currently near-empty in practice.
- Bundle cross-section frames use an arbitrary perpendicular basis; real tools
  align the cross-section to the incident face normal. This affects the 3D
  coordinates of crossovers, not their bp indices or the graph topology.
- Mates are treated as *available* links, not mandatory ones: the scaffold may
  use a vertex connection or leave it to the staples.
- Staple routing is a gate, not a router. A helix stretch longer than
  `MAX_STAPLE_SPAN_BP` (49 bp) with no free crossover fails, but staples are
  never actually laid out. Sequence assignment and real file export are still
  out of scope.
- Each cylinder is traversed by the scaffold exactly once, end to end. Real
  routers may split a helix across several scaffold passes.
