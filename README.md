# CAD-DNA

A benchmark + surrogate for **CAD-level DNA origami routability**.

Existing tools (caDNAno, MagicDNA, ATHENA, DAEDALUS, oxDNA) follow
`design → routing → sequence → simulation`, so you only find out whether a
design is good *after* routing. The goal here is to insert a learned surrogate
before routing:

```
CAD geometry → heterogeneous graph → GNN → risk prediction → route or not
```

## Pipeline

```
CAD params                       cadna/params.py      [done]
   -> Feature / Bundle / Cylinder    cadna/generator.py   [done]
   -> cylinder adjacency             cadna/adjacency.py   [done]
   -> candidate crossovers           cadna/adjacency.py   [done]
   -> mate constraints               cadna/generator.py   [done]
   -> routing precheck               cadna/precheck.py    [next]
   -> scaffold routing (DFS)         cadna/routing.py     [next]
   -> result record                  labels               [next]
   -> graph extraction               cadna/graph.py       [next]
   -> graph + label on disk          dataset              [next]
   -> surrogate model                models/              [next]
```

## Prediction targets

| target | type | source |
|---|---|---|
| routing feasibility | binary | did the router return a valid route |
| Hamilton-cycle availability | binary | does the cylinder/crossover topology admit a Hamiltonian-style route |
| failure diagnosis | multi-class | pairability / Hamilton / geometry / timeout / staple routing / export |
| search cost | regression | DFS node expansions, wall time, timeout flag |

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
scripts/
  gen_designs.py sample N designs into a directory + index.csv
tests/
  test_generator.py
```

## Usage

```bash
pip install -r requirements.txt
python tests/test_generator.py                          # 11 checks, no pytest needed
python scripts/gen_designs.py --n 500 --out data/designs_v0
```

```python
from cadna import CADParams, sample_params, generate, save_design

d = generate(sample_params(seed=42))                    # random point of the space
d = generate(CADParams(shape="polyhedron", polyhedron="cube",
                       lattice="honeycomb", helices_per_edge=2, edge_bp=63))
print(d.summary())
save_design(d, "data/cube.json.gz")
```

## Shape families

| shape | bundles | mates | regime |
|---|---|---|---|
| `brick` | 1 | none | multi-layer block, crossover rich |
| `plate` | 1 | none | 1-2 layer sheet, sparse and long |
| `polygon_ring` | N | vertex | planar wireframe |
| `polyhedron` | 6-12 | vertex | 3D wireframe (tetra / octa / cube), the DAEDALUS/ATHENA regime |

The sampler deliberately emits degenerate designs -- odd helix counts per edge,
punched-out cross-sections, scaffolds far too short -- because those are the
negative labels the surrogate has to learn. In a 300-design batch roughly 17%
overrun the scaffold and 5% have unpairable helix ends before routing is even
attempted.

## Crossover rule

A crossover occupies the **same bp index on both helices**, so the phase offset
must be symmetric in the two lattice sites. `lattice.py` derives it from the
direction of the inter-helix vector:

- honeycomb: 3 neighbour axes 120 deg apart, `step = 21` -> offsets `{0, 7, 14}`
- square: 2 axes 90 deg apart with parity-alternating helix phase, `step = 32`
  -> offsets `{0, 8, 16, 24}`

The tables are isolated in `Lattice.crossover_offset` so they can be swapped for
the exact caDNAno lookup tables without touching anything downstream.

## Known simplifications

- Inter-bundle contacts are detected and stored as graph edges, but crossovers
  are only enumerated for them when `params.inter_bundle_crossovers` is set,
  since two bundles share no bp frame. Wireframe bundles meet at angles, so this
  is currently near-empty in practice.
- Bundle cross-section frames use an arbitrary perpendicular basis; real tools
  align the cross-section to the incident face normal. This affects the 3D
  coordinates of crossovers, not their bp indices or the graph topology.
- Scaffold/staple nick placement, sequence assignment and export are out of
  scope until the routing stage lands.
