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

- Target environment: Python 3.13 with PyTorch, NumPy, NetworkX and
  `torch_geometric`.
- Completed: the whole `CAD -> precheck -> routing -> label -> hetero graph ->
  surrogate` pipeline.
- Dataset: `data/designs_v1` holds 10,000 iid designs + `labels.csv`;
  `data/designs_v1_graphs` the matching hetero graphs (986k nodes, 3.86M
  directed edges). `data/designs_v1_rare` is a rejection-sampled supplement for
  the two thin failure classes, marked so it can only enter training.
  `data/designs_v0` is the earlier 1,000-design run, kept for comparison.
- Labels (10k): 2,407 routable (24.1%); Hamiltonian cycle 4,516 True / 5,207
  False / 277 unknown (budget); Hamiltonian path 7,221 / 2,403 / 376. 6,547
  designs reach the search. Failure classes: `hamilton` 3,658,
  `staple_routing` 1,433, `pairability` 1,290, `scaffold_length` 676,
  `timeout` 277, `geometry` 259, `export` 0.
- The 1k and 10k runs agree closely (24.5% vs 24.1% routable), so the label
  distribution is a property of the parameter space, not of the sample size.
- Results: `data/baseline_v1.json` (iid) and `data/baseline_v1_rare.json` (with
  the oversampled supplement in training). See **The surrogate** below.
- Next milestone: score the models on the subset the precheck cannot decide,
  which is the only part of the distribution where a surrogate earns its keep.

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
   -> surrogate model               models/              [done]
```

## Prediction targets

| target | column | type | source |
|---|---|---|---|
| routing feasibility | `routable` | binary | the whole precheck → route → staple → export chain succeeded |
| Hamilton-cycle availability | `hamilton` | binary, `None` on timeout | a Hamiltonian cycle exists over the port graph |
| …allowing a free scaffold closure | `hamilton_path` | binary, `None` on timeout | a Hamiltonian *path* exists (see below) |
| failure diagnosis | `failure_class` | multi-class | `pairability` / `hamilton` / `geometry` / `timeout` / `staple_routing` / `export` / `scaffold_length` |
| search cost | `nodes_expanded`, `backtracks`, `elapsed_s`, `timeout` | regression | the DFS budget counters |

**Search cost is scored relative to size, and it is censored.** `nodes_expanded`
grows with the cylinder count, so a plain rank correlation against it mostly
measures whether the model noticed how big the design is -- on the 1k set a
predictor using *only* size scores rho = 0.80. `models.data.CostBaseline` fits
a per-size-bin median on the training split, the cost head regresses the
residual against it, and `cost_rho_within` correlates inside size bins, where
the size-only predictor is constant and scores nothing at all. Separately, a
design that hit the node budget reports `nodes_expanded == 200000`, which is a
lower bound: those are excluded from the cost regression and from the cost
metrics, and enter the loss only through a one-sided hinge.

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
  train_baseline.py train and compare the baselines -> baseline_results.json
models/
  data.py        loading, the split, normalisation, the class map, the cost baseline
  metrics.py     balanced accuracy, AUC, macro F1, (stratified) Spearman, MAE
  nets.py        GraphMLP (graph-level features only) and HeteroGNN
  train.py       the multi-task loop, the scoring, and the reference baselines
tests/
  test_generator.py
  test_routing.py
  test_graph.py
  test_models.py
```

## Usage

```bash
pip install -r requirements.txt
for t in tests/test_*.py; do python "$t"; done   # no pytest needed
python scripts/gen_designs.py   --n 1000 --out data/designs_v0
python scripts/route_designs.py --designs data/designs_v0
python scripts/build_graphs.py  --designs data/designs_v0   # -> data/designs_v0_graphs
python scripts/train_baseline.py --graphs data/designs_v0_graphs --seeds 3

# top the thin failure classes up with rejection-sampled designs, marked
# `rare:<class>` so they stay out of validation and test
python scripts/gen_designs.py --n 10000 --out data/designs_v1     --oversample geometry,timeout --oversample-n 300
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
| `feature` | CAD primitives | 15 | kind, length, centre + direction, cross-section params |
| `bundle` | helix groups | 15 | lattice, length in bp and turns, cross-section extent, axis |
| `cylinder` | helices | 27 | bp interval, lattice site, geometry, **port degrees**, link counts |
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

No column is an exact copy of another: `n_features` equalled `n_bundles`,
`n_links_mate` equalled `n_mates`, `port_deg_mean` *is* `link_density` (2L/2n),
`inset_lo_bp` was `bp_start` again, and `length_nm` was `bp_len` times the rise.
They are gone, and `test_no_duplicate_feature_columns` fails if a new one
appears. One-hot blocks keep their usual rank deficiency.

**Two graph-level blocks, deliberately separate.** `graph_x` (21) is pure
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

## The surrogate

`scripts/train_baseline.py` trains four configurations over 3 seeds and reports
test metrics next to the three reference rows. Numbers below: 10,000 iid
designs, 20 epochs, ~1,500 test designs, mean +- sd over seeds.

| model | routable bacc | hamilton bacc | failure macro F1 | cost rho (within size) | cost MAE |
|---|---|---|---|---|---|
| `majority` | 0.500 | 0.500 | 0.077 | -- | -- |
| `precheck-rule` | 0.803 | 0.790 | -- | -- | -- |
| `size-only` | -- | -- | -- | -- | 1.013 |
| `mlp-counts` | 0.944 +- 0.01 | 0.923 +- 0.01 | 0.773 +- 0.01 | 0.767 +- 0.02 | 0.502 |
| `mlp-rules` | 0.963 +- 0.00 | 0.952 +- 0.00 | 0.919 +- 0.01 | 0.818 +- 0.01 | 0.426 |
| `gnn-structure` | 0.973 +- 0.01 | 0.983 +- 0.01 | 0.948 +- 0.00 | 0.865 +- 0.00 | 0.224 |
| `gnn-rules` | **0.984 +- 0.00** | **0.992 +- 0.00** | **0.977 +- 0.00** | **0.870 +- 0.01** | 0.235 |

Per failure class, which is where the macro average stops being informative:

| failure F1 | none | geometry | pairability | hamilton | timeout | staple | scaffold |
|---|---|---|---|---|---|---|---|
| test support | 367 | 40 | 203 | 540 | 44 | 210 | 93 |
| `mlp-counts` | 0.876 | 0.330 | 0.906 | 0.872 | 0.753 | 0.899 | 0.778 |
| `mlp-rules` | 0.915 | 1.000 | 1.000 | 0.935 | 0.792 | 0.919 | 0.873 |
| `gnn-structure` | 0.957 | 0.851 | 0.986 | 0.964 | 0.973 | 0.951 | 0.953 |
| `gnn-rules` | 0.971 | 0.996 | 0.998 | 0.986 | 0.973 | 0.960 | 0.953 |

Three things this says:

1. **`mlp-rules` scoring exactly 1.000 on `geometry` and `pairability` is the
   leakage, not a result.** Those two classes *are* `precheck_x`; handing a
   model the block hands it the answer. This is why the block is separable.
2. **Structure carries what the rules do not.** `gnn-structure` never sees
   `precheck_x` and still reaches 0.851 / 0.986 on those same two classes -- it
   rediscovers the obstructions from the graph. And on `timeout` it goes the
   other way, 0.973 against the rules baseline's 0.792: how expensive a search
   will be is not something an exact obstruction knows anything about.
3. **The cost head is doing more than reading off the size.** A size-only
   predictor gets rho = 0.804 overall and nothing at all within a size bin; the
   GNNs reach 0.87 within bins, and cut the MAE from 1.01 to 0.22.

### Does oversampling the thin classes help?

Same seeds and the same iid test set, adding the 300-design `rare:` supplement
to *training only* (`--graphs data/designs_v1_graphs,data/designs_v1_rare_graphs`):

| failure F1 | `mlp-counts` | `mlp-rules` | `gnn-structure` | `gnn-rules` |
|---|---|---|---|---|
| `geometry` | 0.330 -> **0.489** | 1.000 -> 1.000 | 0.851 -> 0.881 | 0.996 -> 0.996 |
| `timeout` | 0.753 -> 0.733 | 0.792 -> **0.828** | 0.973 -> **0.982** | 0.973 -> 0.971 |
| macro | 0.773 -> 0.795 | 0.919 -> 0.924 | 0.948 -> 0.956 | 0.977 -> 0.976 |

It helps where a model was starved and does nothing where one had already
saturated the class -- 2.4% more training data moved `mlp-counts` on `geometry`
by +0.16 and left `gnn-rules` where it was. Since the supplement cannot enter
the test set, none of this makes the *measurement* of a thin class more precise;
the support column stays at 40 and 44 either way. Buying that would mean a
larger iid draw, not a bigger supplement.

## Sampling and the thin classes

The default sampler is iid over the CAD parameter space, and two failure classes
come out thin: `geometry` (2.6%) and `timeout` (2.8%). `gen_designs.py
--oversample geometry,timeout` tops them up by rejection sampling against a
cheap probe -- the precheck for the classes it decides outright, a 20k-node DFS
as a proxy for `timeout`, since a design that blows a small budget usually blows
the real one.

The probe is exact for `geometry` -- it is a precheck verdict, and no budget
changes it -- and 61% precise for `timeout`: of 150 designs that blew the 20k
probe budget, 92 also blew the real 200k one, while the rest turned out to be
`hamilton` (29), `scaffold_length` (19) or routable (8). Those are kept with
their true labels; a proxy that over-fires is a sampling bias, not a wrong
label. 5,036 draws filled both quotas.

Those designs are not iid, so they are marked `rare:<class>` in the design
notes, the index, and the graph metadata, and `models.data.make_split` puts
anything marked into **training only**. The consequence is worth stating: this
buys the model more examples of a thin class, it does *not* make the test-set
measurement of that class more precise. That would need a distorted test set,
which reports a class balance no real design stream has.

`export` is the opposite case: it has no examples at all and cannot get any,
because a route that fails `validate_route` is a router bug, not a property of a
design. `models.data.ClassMap` therefore gives the failure head one slot per
class the dataset actually contains, and reports what it dropped.

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

`train_baseline.py` prints three reference rows above the models, one per kind
of free lunch:

| reference | what it is | what it makes free |
|---|---|---|
| `majority` | always the training-majority class | class imbalance |
| `precheck-rule` | the pipeline's own fatal-precheck verdict | every exact obstruction |
| `size-only` | the training-fitted cost median for the design's size bin | most of the cost rank correlation |

A model is only interesting where it is above all three. `precheck-rule` is
also why `graph_x` and `precheck_x` are separate blocks: `--only` a config with
`include_precheck=False` is the ablation that asks whether the structure carries
anything the rules do not.

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
