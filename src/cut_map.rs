//! Per-cut genlib cell matching + mapping DP (the ABC `mapperMatch.c`
//! analogue over e-graph cuts).
//!
//! Matching: the library is indexed ONCE by every pin permutation of every
//! cell — key `(arity, tt_of_variant)` where the variant is
//! `permute_tt(cell.tt, n, P)` (truth.permute_tt semantics: variant
//! variable i = cell pin P[i]).  A cut matches when its
//! `(len(leaves), tt)` hits the key; the cell then consumes cut leaf
//! `Pinv[i]` at pin i (Pinv = P inverse — T17 pins this down), keeping the
//! best-area and best-delay entries per key (ABC's `pMBestA`/`pMBestD`).
//!
//! DP (single pass, children-first over the SCC condensation): per class,
//! candidates are its non-trivial cuts matched to a cell; arrival =
//! cell delay + max leaf arrival; area flow = cell area + sum of
//! leaf-flow / reference-estimate (ABC `Map_CutGetAreaFlow`).  Objective
//! "delay" ranks by (arrival, flow), "area" by (flow, arrival); the
//! deterministic FNV-of-inputs breaks remaining ties.  Trivial-cut products
//! already reproduce every e-node's own shape, so the candidate set is
//! complete — no separate op-binding path exists.  A class with no
//! candidate (no cut matches any cell, or a leaf class is unmappable) is
//! marked unemittable and excluded from its consumers; an unemittable PO
//! is a hard error.

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::class_graph::{ClassGraph, pct};
use crate::cut_iter::{CutResult, ClassKind, Kernel, class_kind};

#[derive(Clone, Debug)]
pub(crate) struct CellSpec {
    pub(crate) name: String,
    pub(crate) arity: usize,
    pub(crate) tt: u64,
    pub(crate) area: f64,
    pub(crate) delay: f64,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    cell: usize,
    /// Cell pin i is fed by cut leaf `perm[i]`.
    perm: [u8; 4],
}

/// Index over all pin permutations of all cells.
struct MatchIndex {
    /// (arity, variant tt) -> (best-area entry, best-delay entry).
    by_key: HashMap<(usize, u64), (Option<Entry>, Option<Entry>)>,
}

/// truth.permute_tt semantics: result bit m = tt bit at the minterm whose
/// bit i equals bit perm[i] of m.
fn permute_tt(tt: u64, n: usize, perm: &[u8; 4]) -> u64 {
    let mut out = 0u64;
    for m in 0..(1u64 << n) {
        let mut src = 0usize;
        for i in 0..n {
            if (m >> i) & 1 == 1 {
                src |= 1 << perm[i];
            }
        }
        if (tt >> src) & 1 == 1 {
            out |= 1 << m;
        }
    }
    out
}

fn invert(perm: &[u8; 4], n: usize) -> [u8; 4] {
    let mut inv = [0u8; 4];
    for i in 0..n {
        inv[perm[i] as usize] = i as u8;
    }
    inv
}

fn permutations(n: usize) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    let mut cur: [u8; 4] = [0, 1, 2, 3];
    fn heap(k: usize, cur: &mut [u8; 4], out: &mut Vec<[u8; 4]>) {
        if k == 1 {
            out.push(*cur);
            return;
        }
        for i in 0..k {
            heap(k - 1, cur, out);
            let j = if k % 2 == 0 { i } else { 0 };
            cur.swap(j, k - 1);
        }
    }
    if n > 0 {
        heap(n, &mut cur, &mut out);
    }
    out
}

fn build_index(cells: &[CellSpec]) -> Result<MatchIndex, String> {
    let mut by_key: HashMap<(usize, u64), (Option<Entry>, Option<Entry>)> = HashMap::new();
    // cells arrive sorted by name from the caller: deterministic insertion
    for (ci, c) in cells.iter().enumerate() {
        if c.arity == 0 || c.arity > 4 {
            return Err(format!(
                "cell {:?} arity {} unsupported (1..=4)",
                c.name, c.arity
            ));
        }
        for p in permutations(c.arity) {
            // variant variable i = cell pin p[i]; variant == cut tt means
            // cell pin i is fed by cut leaf p^{-1}[i]
            let key = (c.arity, permute_tt(c.tt, c.arity, &p));
            let e = Entry {
                cell: ci,
                perm: invert(&p, c.arity),
            };
            let slot = by_key.entry(key).or_insert((None, None));
            let ba = &mut slot.0;
            let better_area = match ba {
                None => true,
                Some(cur) => {
                    let (ca, cd) = (cells[cur.cell].area, cells[cur.cell].delay);
                    (c.area, c.delay, e.perm) < (ca, cd, cur.perm)
                }
            };
            if better_area {
                *ba = Some(e);
            }
            let bd = &mut slot.1;
            let better_delay = match bd {
                None => true,
                Some(cur) => {
                    let (ca, cd) = (cells[cur.cell].area, cells[cur.cell].delay);
                    (c.delay, c.area, e.perm) < (cd, ca, cur.perm)
                }
            };
            if better_delay {
                *bd = Some(e);
            }
        }
    }
    Ok(MatchIndex { by_key })
}

#[derive(Clone, Debug)]
struct Choice {
    /// cut index within the class's cut list.
    cut: usize,
    cell: usize,
    perm: [u8; 4],
    /// Class ids in cell-pin order.
    inputs: Vec<u32>,
    arrival: f64,
    flow: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MapStats {
    pub(crate) num_classes: usize,
    pub(crate) num_cone_classes: usize,
    pub(crate) num_mapped: usize,
    pub(crate) num_unmatched: usize,
    pub(crate) num_pi: usize,
    pub(crate) num_const: usize,
    pub(crate) max_root_arrival: f64,
    pub(crate) total_flow_at_roots: f64,
}

pub(crate) struct MapOutput {
    pub(crate) arrival: Vec<f64>,
    pub(crate) flow: Vec<f64>,
    pub(crate) choices: Vec<Option<Choice>>,
    pub(crate) emit_ok: Vec<bool>,
    pub(crate) stats: MapStats,
}

fn fnv_inputs(inputs: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for l in inputs {
        for b in l.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Single-pass mapping DP over the cut sets.
pub(crate) fn map_cuts(
    graph: &ClassGraph,
    result: &CutResult,
    kernels: &[Kernel],
    cone: &[bool],
    roots: &[u32],
    cells: &[CellSpec],
    objective: &str,
) -> Result<MapOutput, String> {
    if objective != "delay" && objective != "area" {
        return Err(format!(
            "objective must be 'delay' or 'area', got {objective:?}"
        ));
    }
    let index = build_index(cells)?;
    let n = graph.classes.len();
    let cuts = &result.cuts;

    // reference estimates: e-node child references over the cone
    let mut refs = vec![0u32; n];
    for (v, ok) in cone.iter().enumerate() {
        if *ok {
            for &ei in &graph.enodes_of[v] {
                for &c in &graph.enodes[ei].ch_eq {
                    refs[c as usize] += 1;
                }
            }
        }
    }

    let mut arrival = vec![f64::INFINITY; n];
    let mut flow = vec![f64::INFINITY; n];
    let mut choices: Vec<Option<Choice>> = vec![None; n];
    let mut emit_ok = vec![false; n];
    let mut const_classes_tmp: HashMap<u32, u64> = HashMap::new();

    let mut stats = MapStats {
        num_classes: n,
        num_cone_classes: cone.iter().filter(|&&b| b).count(),
        ..Default::default()
    };

    let delay_first = objective == "delay";

    for (si, comp) in graph.sccs.iter().enumerate() {
        // Complement pairs (c, ~c) make SCC members reference each other's
        // choices; like the cut iterator, sweep the SCC by FIXPOINT
        // ITERATION, deferring classes whose candidate leaves are not yet
        // emittable, until no progress.  Leftovers are unmatched.
        let scc_pos: HashMap<u32, usize> =
            comp.iter().enumerate().map(|(i, &cv)| (cv, i)).collect();
        let mut done: Vec<bool> = vec![false; comp.len()];
        for _pass in 0..=comp.len() {
            let mut progress = false;
            for i in 0..comp.len() {
                if done[i] {
                    continue;
                }
                let cv = comp[i];
                let v = cv as usize;
                if !cone[v] {
                    done[i] = true;
                    continue;
                }
                let kind = class_kind(graph, kernels, v).map_err(|e| e.to_string())?;
                match kind {
                    ClassKind::Pi => {
                        stats.num_pi += 1;
                        arrival[v] = 0.0;
                        flow[v] = 0.0;
                        emit_ok[v] = true;
                        done[i] = true;
                        progress = true;
                        continue;
                    }
                    ClassKind::Const(c) => {
                        stats.num_const += 1;
                        arrival[v] = 0.0;
                        flow[v] = 0.0;
                        emit_ok[v] = true;
                        const_classes_tmp.insert(cv, c);
                        done[i] = true;
                        progress = true;
                        continue;
                    }
                    ClassKind::Internal => {}
                }

                let mut best: Option<(f64, f64, u64, Choice)> = None;
                let mut consider = |best: &mut Option<(f64, f64, u64, Choice)>,
                                    arr: f64,
                                    fl: f64,
                                    ch: Choice| {
                    let key = if delay_first {
                        (arr, fl, fnv_inputs(&ch.inputs))
                    } else {
                        (fl, arr, fnv_inputs(&ch.inputs))
                    };
                    let replace = match best {
                        None => true,
                        Some((ba, bf, bh, _)) => {
                            let cur_key = if delay_first {
                                (*ba, *bf, *bh)
                            } else {
                                (*bf, *ba, *bh)
                            };
                            key < cur_key
                        }
                    };
                    if replace {
                        *best = Some((arr, fl, fnv_inputs(&ch.inputs), ch));
                    }
                };
                for (ci, c) in cuts[v].iter().enumerate() {
                    if c.realization.is_none() {
                        continue; // trivial cut: not cell-matchable
                    }
                    let arity = c.leaves.len();
                    if arity == 0 || arity > 4 {
                        continue;
                    }
                    let Some((ba, bd)) = index.by_key.get(&(arity, c.tt)) else {
                        continue;
                    };
                    for entry in [ba, bd].into_iter().flatten() {
                        let inputs: Vec<u32> = (0..arity)
                            .map(|i| c.leaves[entry.perm[i] as usize])
                            .collect();
                        if inputs.iter().any(|&i| !emit_ok[i as usize]) {
                            continue;
                        }
                        let cell = &cells[entry.cell];
                        let arr = cell.delay
                            + inputs
                                .iter()
                                .map(|&i| arrival[i as usize])
                                .fold(f64::NEG_INFINITY, f64::max);
                        let fl = cell.area
                            + inputs
                                .iter()
                                .map(|&i| flow[i as usize] / (refs[i as usize].max(1) as f64))
                                .sum::<f64>();
                        consider(
                            &mut best,
                            arr,
                            fl,
                            Choice {
                                cut: ci,
                                cell: entry.cell,
                                perm: entry.perm,
                                inputs,
                                arrival: arr,
                                flow: fl,
                            },
                        );
                    }
                }
                match best {
                    Some((arr, fl, _, ch)) => {
                        arrival[v] = arr;
                        flow[v] = fl;
                        choices[v] = Some(ch);
                        emit_ok[v] = true;
                        stats.num_mapped += 1;
                        done[i] = true;
                        progress = true;
                    }
                    None => {
                        // deferred: a same-SCC leaf may become emittable in a
                        // later pass; if the sweep stalls this becomes unmatched
                    }
                }
            }
            if !progress {
                break;
            }
        }
        for i in 0..comp.len() {
            if done[i] {
                continue;
            }
            done[i] = true;
            let v = comp[i] as usize;
            if cone[v] {
                stats.num_unmatched += 1;
            }
        }
    }

    let mut max_root_arrival = 0.0f64;
    let mut total_flow_at_roots = 0.0f64;
    for &r in roots {
        let r = r as usize;
        if !emit_ok[r] {
            let bad = roots.iter().filter(|&&x| !emit_ok[x as usize]).count();
            return Err(format!(
                "cut-map: {bad} of {} root classes are unmappable with this \
                 library/K (unmatched classes: {}); raise K/max_cuts or use \
                 the hybrid mapper",
                roots.len(),
                stats.num_unmatched
            ));
        }
        max_root_arrival = max_root_arrival.max(arrival[r]);
        total_flow_at_roots += flow[r];
    }
    stats.max_root_arrival = max_root_arrival;
    stats.total_flow_at_roots = total_flow_at_roots;
    Ok(MapOutput {
        arrival,
        flow,
        choices,
        emit_ok,
        stats,
    })
}

/// Mapping decisions over the cut sets of a [`crate::cut_iter::CutIterator`].
#[pyclass(unsendable)]
pub struct CutMapper {
    choices: Vec<Option<Choice>>,
    emit_ok: Vec<bool>,
    cells: Vec<CellSpec>,
    stats: MapStats,
    const_classes: HashMap<u32, u64>,
    pi_names: HashMap<u32, String>,
}

#[pymethods]
impl CutMapper {
    /// Run the mapping DP over `cut_iterator`'s enumerated cuts.
    ///
    /// `cells`: list of (name, arity, tt, area, delay) tuples — build with
    /// `eggverse.cutmap.cell_table(library)`; delay is the cell head delay
    /// at the DP's nominal load (matches the existing cost-model convention).
    #[new]
    #[pyo3(signature = (cut_iterator, cells, objective = "delay"))]
    fn new(
        cut_iterator: PyRef<'_, crate::cut_iter::CutIterator>,
        cells: Vec<(String, usize, u64, f64, f64)>,
        objective: &str,
    ) -> PyResult<Self> {
        if cells.is_empty() {
            return Err(PyValueError::new_err("cell table must not be empty"));
        }
        let mut cells: Vec<CellSpec> = cells
            .into_iter()
            .map(|(name, arity, tt, area, delay)| CellSpec {
                name,
                arity,
                tt,
                area,
                delay,
            })
            .collect();
        cells.sort_by(|a, b| a.name.cmp(&b.name));
        let cit = &*cut_iterator;
        let out = map_cuts(
            &cit.graph,
            &cit.result,
            &cit.kernels,
            &cit.cone,
            &cit.root_ids,
            &cells,
            objective,
        )
        .map_err(PyValueError::new_err)?;
        // emittable classes without a Choice are PIs or constants
        let mut const_classes = HashMap::new();
        for (v, ch) in out.choices.iter().enumerate() {
            if out.emit_ok[v] && ch.is_none() {
                let kind = class_kind(&cit.graph, &cit.kernels, v)
                    .map_err(PyValueError::new_err)?;
                if let ClassKind::Const(c) = kind {
                    const_classes.insert(v as u32, c);
                }
            }
        }
        Ok(CutMapper {
            choices: out.choices,
            emit_ok: out.emit_ok,
            cells,
            stats: out.stats,
            const_classes,
            pi_names: cit.pi_names.clone(),
        })
    }

    /// Mapping decisions per class:
    /// {class_id: {"cell": name, "inputs": [class ids in cell-pin order],
    /// "cut": cut idx, "arrival": f, "flow": f}}.
    #[getter]
    fn decisions<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new(py);
        for (v, ch) in self.choices.iter().enumerate() {
            let Some(ch) = ch else { continue };
            let e = pyo3::types::PyDict::new(py);
            e.set_item("cell", self.cells[ch.cell].name.clone())?;
            e.set_item("inputs", ch.inputs.clone())?;
            e.set_item("cut", ch.cut)?;
            e.set_item("arrival", ch.arrival)?;
            e.set_item("flow", ch.flow)?;
            d.set_item(v, e)?;
        }
        Ok(d.into_any().unbind())
    }

    /// Cone classes that cannot be emitted (no cut matched the library).
    #[getter]
    fn unmatched_classes<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let l = pyo3::types::PyList::empty(py);
        for (v, ok) in self.emit_ok.iter().enumerate() {
            if !ok {
                l.append(v)?;
            }
        }
        Ok(l.into_any().unbind())
    }

    /// PI classes: {class_id: PI name (the var primitive)}.
    #[getter]
    fn pi_names<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new(py);
        for (k, v) in &self.pi_names {
            d.set_item(k, v)?;
        }
        Ok(d.into_any().unbind())
    }

    /// Constant classes: {class_id: 0 | 1}.
    #[getter]
    fn const_classes<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new(py);
        for (k, v) in &self.const_classes {
            d.set_item(k, v)?;
        }
        Ok(d.into_any().unbind())
    }

    /// Mapping statistics.
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let s = &self.stats;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("num_classes", s.num_classes)?;
        d.set_item("num_cone_classes", s.num_cone_classes)?;
        d.set_item("num_mapped", s.num_mapped)?;
        d.set_item("num_unmatched", s.num_unmatched)?;
        d.set_item("num_pi", s.num_pi)?;
        d.set_item("num_const", s.num_const)?;
        d.set_item("max_root_arrival", s.max_root_arrival)?;
        d.set_item("total_flow_at_roots", s.total_flow_at_roots)?;
        d.set_item("unmatched_pct", pct(s.num_unmatched, s.num_cone_classes))?;
        Ok(d.into_any().unbind())
    }
}

// -------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_graph::tarjan_scc;
    use crate::cut_iter::fixtures::{leaf_leaves, mkgraph, params};
    use crate::cut_iter::{CutParams, enumerate_cuts};

    use crate::class_graph::Enode;
    use egglog::Value as EggValue;

    /// var e-nodes (mkgraph only builds eq-children enodes; PI classes need
    /// their `var` rows to be recognized as PIs).
    fn add_vars(g: &mut crate::class_graph::ClassGraph, var_func: usize, classes: &[usize]) {
        for (j, &c) in classes.iter().enumerate() {
            g.enodes.push(Enode {
                func: var_func,
                out: c as u32,
                ch_eq: vec![],
                ch_prim: vec![(0, EggValue::new_const(j as u32 + 100))],
                head: 0.0,
            });
            g.enodes_of[c].push(g.enodes.len() - 1);
        }
    }

    fn cell(name: &str, arity: usize, tt: u64, area: f64, delay: f64) -> CellSpec {
        CellSpec {
            name: name.to_string(),
            arity,
            tt,
            area,
            delay,
        }
    }

    fn std_cells() -> Vec<CellSpec> {
        vec![
            cell("and2", 2, 0b1000, 3.0, 2.0),
            cell("nand2", 2, 0b0111, 2.0, 1.5),
            cell("or2", 2, 0b1110, 2.0, 2.0),
            cell("inv", 1, 0b01, 1.0, 1.0),
        ]
    }

    /// T14: delay-first picks NAND2 for the sound choice class
    /// {or(nx,ny), nand(x,y)} (demorgan twin) and reports cell-pin-order
    /// inputs.
    #[test]
    fn t14_delay_first_choice() {
        // 0=x,1=y,2=nx,4=ny, 3 = choice{or(2,4), nand(0,1)}
        let mut g = mkgraph(
            5,
            vec![
                ("var", vec![false]),
                ("B_not", vec![true]),
                ("B_or", vec![true, true]),
                ("B_nand", vec![true, true]),
            ],
            vec![
                (1, 2, vec![0]),
                (1, 4, vec![1]),
                (2, 3, vec![2, 4]),
                (3, 3, vec![0, 1]),
            ],
        );
        add_vars(&mut g, 0, &[0, 1]);
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt_of("not") },
            Kernel::Op { tt: op_tt_of("or") },
            Kernel::Op { tt: op_tt_of("nand") },
        ];
        let cone = vec![true; 5];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let roots = [3u32];
        let cells = std_cells();
        let out = map_cuts(&g, &res, &kernels, &cone, &roots, &cells, "delay").unwrap();
        let ch = out.choices[3].as_ref().unwrap();
        assert_eq!(cells[ch.cell].name, "nand2");
        assert_eq!(ch.arrival, 1.5);
        assert_eq!(ch.inputs, vec![0, 1]);
        assert!(out.emit_ok.iter().all(|&b| b));
    }

    /// T15: area objective prefers the smaller area-flow (or2 over two cheap
    /// inverters beats the direct nand2).
    #[test]
    fn t15_area_objective() {
        let mut cells = std_cells();
        cells[2] = cell("or2", 2, 0b1110, 1.0, 2.0);
        cells[3] = cell("inv", 1, 0b01, 0.25, 1.0);
        let mut g = mkgraph(
            5,
            vec![
                ("var", vec![false]),
                ("B_not", vec![true]),
                ("B_or", vec![true, true]),
                ("B_nand", vec![true, true]),
            ],
            vec![
                (1, 2, vec![0]),
                (1, 4, vec![1]),
                (2, 3, vec![2, 4]),
                (3, 3, vec![0, 1]),
            ],
        );
        add_vars(&mut g, 0, &[0, 1]);
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt_of("not") },
            Kernel::Op { tt: op_tt_of("or") },
            Kernel::Op { tt: op_tt_of("nand") },
        ];
        let cone = vec![true; 5];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let roots = [3u32];
        let out = map_cuts(&g, &res, &kernels, &cone, &roots, &cells, "area").unwrap();
        let ch = out.choices[3].as_ref().unwrap();
        assert_eq!(cells[ch.cell].name, "or2");
        assert_eq!(ch.inputs, vec![2, 4]);
        // flow = or2 area 1.0 + inv flow 0.25 (ref=1) + inv flow 0.25
        assert!((ch.flow - 1.5).abs() < 1e-9);
    }

    /// T16: unmatched class filtered from consumers; unmappable root is a hard
    /// error.
    #[test]
    fn t16_unmatched_filter_and_po_error() {
        // 0=x,1=y, 2=or(x,y) (unmatchable with and-only library), 3=and(2,x)
        let mut g = mkgraph(
            4,
            vec![
                ("var", vec![false]),
                ("B_or", vec![true, true]),
                ("B_and", vec![true, true]),
            ],
            vec![(1, 2, vec![0, 1]), (2, 3, vec![2, 0])],
        );
        add_vars(&mut g, 0, &[0, 1]);
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt_of("or") },
            Kernel::Op { tt: op_tt_of("and") },
        ];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let cells = vec![cell("and2", 2, 0b1000, 3.0, 2.0)];
        // no root: no error, but 2 and 3 are unmatched
        let out = map_cuts(&g, &res, &kernels, &cone, &[], &cells, "delay").unwrap();
        assert!(!out.emit_ok[2] && !out.emit_ok[3]);
        assert_eq!(out.stats.num_unmatched, 2);
        // with 3 as root: hard error
        let err = map_cuts(&g, &res, &kernels, &cone, &[3u32], &cells, "delay");
        assert!(err.is_err(), "unmappable root must be a hard error");
    }

    /// T17: MUX2 with pins (A,B,S) matches the mux(s,a,b) cut under a pin
    /// permutation, and inputs come back in CELL-PIN order (a, b, s).
    #[test]
    fn t17_mux_pin_permutation() {
        // mux op tt: s=var0, a=var1, b=var2 -> (vm0&vm1)|(~vm0&vm2)
        // MUX2 cell tt over pins (A,B,S): (vm2&vm0)|(~vm2&vm1)
        let vm = |i: usize| crate::cut_iter::fixtures::var_mask_pub(i, 3);
        let mux_cell_tt = (vm(2) & vm(0)) | (!vm(2) & vm(1));
        // 0=s,1=a,2=b, 3=mux(s,a,b)
        let mut g = mkgraph(
            4,
            vec![("var", vec![false]), ("B_mux", vec![true, true, true])],
            vec![(1, 3, vec![0, 1, 2])],
        );
        add_vars(&mut g, 0, &[0, 1, 2]);
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt_of("mux") }];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let cells = vec![cell("mux2", 3, mux_cell_tt, 4.0, 3.0)];
        let roots = [3u32];
        let out = map_cuts(&g, &res, &kernels, &cone, &roots, &cells, "delay").unwrap();
        let ch = out.choices[3].as_ref().unwrap();
        // cell pins (A,B,S) receive (a, b, s) = leaves 1, 2, 0
        assert_eq!(ch.inputs, vec![1, 2, 0]);
        assert_eq!(ch.arrival, 3.0);
    }

    /// T17b: index determinism + best-delay vs best-area selection per key.
    #[test]
    fn t17b_index_best_entries() {
        let cells = std_cells();
        let idx = build_index(&cells).unwrap();
        // (2, 0b0111): nand2 identity — best delay AND best area (only cell)
        let (ba, bd) = idx.by_key.get(&(2, 0b0111)).unwrap();
        assert_eq!(cells[ba.unwrap().cell].name, "nand2");
        assert_eq!(cells[bd.unwrap().cell].name, "nand2");
        // two runs identical
        let idx2 = build_index(&cells).unwrap();
        assert_eq!(idx.by_key.len(), idx2.by_key.len());
    }

    /// T20: DP-side fixpoint: a complement-pair SCC where the ~class has no
    /// candidates until its partner is emitted gets deferred, not dropped.
    #[test]
    fn t20_dp_complement_pair() {
        use crate::cut_iter::ClassKind;
        // T19's graph: 0 = x (var + not(1)), 1 = not(0)
        let mut g = crate::cut_iter::fixtures::mkgraph(
            2,
            vec![("var", vec![false]), ("B_not", vec![true])],
            vec![(1, 1, vec![0]), (1, 0, vec![1])],
        );
        g.enodes.push(Enode {
            func: 0,
            out: 0,
            ch_eq: vec![],
            ch_prim: vec![(0, EggValue::new_const(7))],
            head: 0.0,
        });
        g.enodes_of[0].push(g.enodes.len() - 1);
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt_of("not") }];
        let cone = vec![true; 2];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let cells = vec![cell("inv", 1, 0b01, 1.0, 1.0)];
        let roots = [1u32];
        let out = map_cuts(&g, &res, &kernels, &cone, &roots, &cells, "delay").unwrap();
        assert!(out.emit_ok[0] && out.emit_ok[1]);
        let ch = out.choices[1].as_ref().unwrap();
        assert_eq!(cells[ch.cell].name, "inv");
        assert_eq!(ch.inputs, vec![0]);
        assert_eq!(out.stats.num_unmatched, 0);
        assert_eq!(out.stats.num_pi, 1);
        let _ = ClassKind::Internal;
    }

    fn op_tt_of(op: &str) -> u64 {
        crate::cut_iter::op_tt(op).unwrap()
    }
}
