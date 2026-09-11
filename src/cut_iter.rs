//! ABC-mapper-style cut enumeration over the e-class graph.
//!
//! Port of the structural half of ABC's `map` pipeline
//! (`src/map/mapper/mapperCut.c`) to the e-graph: per e-class cut sets are
//! the union over the class's e-nodes of the merges of its children's cut
//! sets (Cartesian product, leaf-union, size <= K), deduplicated by the
//! sorted leaf array, dominance-filtered (a cut whose leaf set strictly
//! contains a kept cut's leaves is dropped), ranked and capped.  Enumeration
//! is purely STRUCTURAL: operator semantics appear only in the truth-table
//! composition kernel (the one AIG-specific slot in ABC's `Map_TruthsCutOne`),
//! which is a per-op truth table ported 1:1 from `eggverse.cost.operator_tt`.
//!
//! E-graph-specific adaptations:
//! * choices: a class's cut set unions the merges of ALL its e-nodes (ABC
//!   emulates this with `pNextE` choice chains / `Map_CutUnionLists`).  Two
//!   e-nodes realizing the SAME leaf set may yield different free-variable
//!   truth tables when the leaves are correlated (e.g. a class and its
//!   complement both in the cut): the realizations agree only on reachable
//!   leaf assignments.  As in ABC (dedupe keeps the first realization), we
//!   keep the first TT and count the conflicts in `stats.tt_conflicts`;
//!   the composed netlist stays correct because the divergence lives entirely
//!   in unreachable leaf combinations;
//! * cycles: an e-node is enumerable iff every eq child lies in a strictly
//!   earlier SCC (Tarjan components are children-first).  A class with no
//!   enumerable e-node keeps only its trivial cut, i.e. upstream consumers
//!   treat it as a leaf;
//! * phases: negation is an explicit arity-1 `not` e-node here; demorgan
//!   saturation materializes both polarities as sibling classes, which is
//!   the structural analogue of ABC's two-phase `M[0]/M[1]` matching.
//!
//! Every cut's truth table is expressed over the cut's OWN leaves (leaf i =
//! variable i), so it is directly matchable against library cells later.

use std::collections::HashMap;
use std::time::Instant;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::class_graph::ClassGraph;
use crate::{egraph::EGraph, egraph::Value, termdag::TermDag};
use egglog::TermId;
use egglog::ast::Literal;

// ---------------------------------------------------------------- TT helpers
// Convention (mirrors eggverse.truth): for a function over n variables,
// bit m of the mask is the value at the minterm where pin i = (m >> i) & 1.

fn var_mask(i: usize, n: usize) -> u64 {
    let mut m = 0u64;
    for idx in 0..(1usize << n) {
        if (idx >> i) & 1 == 1 {
            m |= 1 << idx;
        }
    }
    m
}

fn full(n: usize) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << (1u64 << n)) - 1
    }
}

fn not_(tt: u64, n: usize) -> u64 {
    !tt & full(n)
}

/// Arity of an eggverse operator (mirrors `eggverse.cost.OP_INFO`).
pub(crate) fn op_arity(op: &str) -> Option<usize> {
    Some(match op {
        "var" | "not" => 1,
        "zero" | "one" => 0,
        "and" | "or" | "xor" | "nand" | "nor" | "xnor" => 2,
        "aoi21" | "oai21" | "mux" => 3,
        "aoi22" | "oai22" => 4,
        _ => return None,
    })
}

/// Truth table of an operator over its own inputs, input i = bit i of the
/// minterm index (mirrors `eggverse.cost.operator_tt`).
pub(crate) fn op_tt(op: &str) -> Option<u64> {
    let (a, b) = (var_mask(0, 2), var_mask(1, 2));
    Some(match op {
        "not" => not_(var_mask(0, 1), 1),
        "and" => a & b,
        "or" => a | b,
        "xor" => a ^ b,
        "nand" => not_(a & b, 2),
        "nor" => not_(a | b, 2),
        "xnor" => not_(a ^ b, 2),
        // aoi21(a,b,c) = !((a&b) | c)
        "aoi21" => not_((var_mask(0, 3) & var_mask(1, 3)) | var_mask(2, 3), 3),
        // aoi22(a,b,c,d) = !((a&b) | (c&d))
        "aoi22" => not_((var_mask(0, 4) & var_mask(1, 4)) | (var_mask(2, 4) & var_mask(3, 4)), 4),
        // oai21(a,b,c) = !((a|b) & c)
        "oai21" => not_((var_mask(0, 3) | var_mask(1, 3)) & var_mask(2, 3), 3),
        // oai22(a,b,c,d) = !((a|b) & (c|d))
        "oai22" => not_((var_mask(0, 4) | var_mask(1, 4)) & (var_mask(2, 4) | var_mask(3, 4)), 4),
        // mux(s,a,b) = (s&a) | (~s&b)
        "mux" => (var_mask(0, 3) & var_mask(1, 3)) | (not_(var_mask(0, 3), 3) & var_mask(2, 3)),
        _ => return None,
    })
}

const VAR1: u64 = 0b10; // var_mask(0, 1): a single free variable

// ------------------------------------------------------------------ kernels

/// Per-constructor semantics for cut enumeration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kernel {
    /// PI constructor (arity 1, one String primitive child): its class is a
    /// primary input; the trivial cut is the variable itself.
    Leaf,
    /// Nullary constant constructor: truth table is the single bit.
    Const(u64),
    /// Boolean operator with a per-op truth table over its eq children.
    Op { tt: u64 },
}

fn build_kernels(graph: &ClassGraph, op_map: &HashMap<String, String>) -> PyResult<Vec<Kernel>> {
    let mut kernels = Vec::with_capacity(graph.funcs.len());
    for f in &graph.funcs {
        let Some(op) = op_map.get(&f.term_name) else {
            return Err(PyValueError::new_err(format!(
                "whitelisted constructor {:?} has no operator mapping",
                f.term_name
            )));
        };
        let n_eq = f.eq_mask.iter().filter(|&&b| b).count();
        let bad = |why: String| {
            PyValueError::new_err(format!(
                "constructor {:?} (op {op:?}) does not match operator shape: {why}",
                f.term_name
            ))
        };
        let k = match op.as_str() {
            "var" => {
                if f.arity != 1 || n_eq != 0 {
                    return Err(bad("var must have exactly one String child".into()));
                }
                Kernel::Leaf
            }
            "zero" => {
                if f.arity != 0 {
                    return Err(bad("zero must be nullary".into()));
                }
                Kernel::Const(0)
            }
            "one" => {
                if f.arity != 0 {
                    return Err(bad("one must be nullary".into()));
                }
                Kernel::Const(1)
            }
            other => {
                let Some(tt) = op_tt(other) else {
                    return Err(PyValueError::new_err(format!(
                        "unknown operator {other:?} for constructor {:?}",
                        f.term_name
                    )));
                };
                let ar = op_arity(other).expect("op_tt implies arity");
                if f.arity != ar || n_eq != ar {
                    return Err(bad(format!("expected {ar} eq children")));
                }
                Kernel::Op { tt }
            }
        };
        kernels.push(k);
    }
    Ok(kernels)
}

/// What a class IS, given the kernels of its e-nodes.  Constant wins over
/// variable: a class with a `zero`/`one` e-node is the constant (an upstream
/// cut must constant-fold through it, not treat it as a free variable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ClassKind {
    Pi,
    Const(u64),
    Internal,
}

pub(crate) fn class_kind(graph: &ClassGraph, kernels: &[Kernel], v: usize) -> PyResult<ClassKind> {
    let mut kind = ClassKind::Internal;
    for &ei in &graph.enodes_of[v] {
        match kernels[graph.enodes[ei].func] {
            Kernel::Leaf => return Ok(ClassKind::Pi),
            Kernel::Const(c) => match kind {
                ClassKind::Const(prev) if prev != c => {
                    return Err(PyValueError::new_err(format!(
                        "unsound e-graph: class {v} has both const-0 and const-1 enodes"
                    )))
                }
                _ => kind = ClassKind::Const(c),
            },
            Kernel::Op { .. } => {}
        }
    }
    Ok(kind)
}

// --------------------------------------------------------------------- cuts

pub(crate) struct CutParams {
    /// Max cut leaves (ABC `nVarsMax`; library-derived there, K=5 in `map`).
    pub k: usize,
    /// Max surviving non-trivial cuts per class (ABC `MAP_CUTS_MAX_USE`-1).
    pub max_cuts: usize,
    /// Consider only the first N enodes of each class (row order).
    pub enode_cap: Option<usize>,
    /// Cap on deduplicated candidates per class before the O(n^2) dominance
    /// filter (ABC effectively works near 250; e-graphs dedupe far less than
    /// AIGs, so this is the main cost lever).
    pub cand_cap: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Cut {
    /// Sorted class indices (canonical key, like ABC's `Num`-sorted leaves).
    pub leaves: Vec<u32>,
    /// Function of the class over the cut leaves, leaf i = variable i.
    pub tt: u64,
    /// (enode idx, per-eq-child cut index) realizing this cut; `None` for
    /// trivial cuts (the class itself as an opaque leaf / constant).
    pub realization: Option<(usize, Vec<u32>)>,
    /// Internal-node count of the realizing cone (tree-counted volume, for
    /// area-flow style ranking later).
    pub nnodes: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct CutStats {
    pub k: usize,
    pub max_cuts: usize,
    pub num_classes_graph: usize,
    pub num_classes_cone: usize,
    pub num_pi_classes: usize,
    pub num_const_classes: usize,
    /// Cone classes with only the trivial cut (PIs, constants, cyclic-only,
    /// no-enode classes).
    pub num_trivial_only: usize,
    pub num_cyclic_classes: usize,
    pub total_nontrivial_cuts: usize,
    /// Same-leafset cuts whose realizing e-nodes disagree on the free-variable
    /// truth table (leaf-correlation don't-cares; first realization kept).
    pub tt_conflicts: usize,
    pub max_cuts_per_class: usize,
    /// (cuts per class, number of classes) histogram, descending by count.
    pub hist: Vec<(usize, usize)>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug)]
pub(crate) struct CutResult {
    /// Per class index (empty for classes outside the cone).
    pub cuts: Vec<Vec<Cut>>,
    pub stats: CutStats,
}

/// Cap on merged candidates considered per e-node (ABC's
/// `MAP_CUTS_MAX_COMPUTE` = 1000 applies per AIG node; a class unions many
/// e-nodes, so per-e-node we spend a fraction of that budget).
const MERGE_BUDGET: usize = 200;

fn fnv_leaves(leaves: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for l in leaves {
        for b in l.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn is_subset(small: &[u32], big: &[u32]) -> bool {
    small.iter().all(|x| big.binary_search(x).is_ok())
}

/// Truth table of `op` over the realizing child cuts, expressed over
/// `parent_leaves` (sorted).  For each parent minterm, each child cut's
/// minterm index is rebuilt from the positions of its leaves in the parent
/// leaf order, then the op truth table selects the output bit.
fn compose_tt(op_tt: u64, children: &[&Cut], parent_leaves: &[u32]) -> u64 {
    let kp = parent_leaves.len();
    // Child j: position of each of its leaves within the parent leaf order.
    let child_pos: Vec<Vec<usize>> = children
        .iter()
        .map(|c| {
            c.leaves
                .iter()
                .map(|l| {
                    parent_leaves
                        .iter()
                        .position(|&x| x == *l)
                        .expect("child leaf must appear in parent leaves")
                })
                .collect()
        })
        .collect();
    let mut out = 0u64;
    for m in 0..(1u64 << kp) {
        let mut idx = 0usize;
        for (j, c) in children.iter().enumerate() {
            let mut mc = 0usize;
            for (i, &p) in child_pos[j].iter().enumerate() {
                if (m >> p) & 1 == 1 {
                    mc |= 1 << i;
                }
            }
            if (c.tt >> mc) & 1 == 1 {
                idx |= 1 << j;
            }
        }
        if (op_tt >> idx) & 1 == 1 {
            out |= 1 << m;
        }
    }
    out
}

/// Enumerate all leaf-union candidates of one e-node: DFS over the Cartesian
/// product of the children's cut lists with incremental size-K pruning.
/// Returns (merged sorted leaves, per-eq-child cut index) pairs.
fn merge_product(lists: &[&[Cut]], k: usize, budget: &mut usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut acc: Vec<u32> = Vec::new();
    let mut chosen: Vec<u32> = Vec::new();
    let mut out: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    dfs(0, lists, k, budget, &mut acc, &mut chosen, &mut out);
    out
}

fn dfs(
    depth: usize,
    lists: &[&[Cut]],
    k: usize,
    budget: &mut usize,
    acc: &mut Vec<u32>,
    chosen: &mut Vec<u32>,
    out: &mut Vec<(Vec<u32>, Vec<u32>)>,
) {
    if *budget == 0 {
        return;
    }
    if depth == lists.len() {
        *budget -= 1;
        out.push((acc.clone(), chosen.clone()));
        return;
    }
    for (ci, c) in lists[depth].iter().enumerate() {
        if *budget == 0 {
            return;
        }
        // incremental sorted union; skip if it would exceed K
        let mut merged: Vec<u32> = Vec::with_capacity(acc.len() + c.leaves.len());
        let (a, b) = (acc.iter().copied(), c.leaves.iter().copied());
        let mut it_a = a.peekable();
        let mut it_b = b.peekable();
        loop {
            match (it_a.peek(), it_b.peek()) {
                (Some(&x), Some(&y)) => {
                    if x < y {
                        merged.push(x);
                        it_a.next();
                    } else if x > y {
                        merged.push(y);
                        it_b.next();
                    } else {
                        merged.push(x);
                        it_a.next();
                        it_b.next();
                    }
                }
                (Some(_), None) => {
                    merged.extend(&mut it_a);
                }
                (None, Some(_)) => {
                    merged.extend(&mut it_b);
                }
                (None, None) => break,
            }
        }
        if merged.len() > k {
            continue;
        }
        let old = std::mem::replace(acc, merged);
        chosen.push(ci as u32);
        dfs(depth + 1, lists, k, budget, acc, chosen, out);
        chosen.pop();
        *acc = old;
    }
}

pub(crate) fn enumerate_cuts(
    graph: &ClassGraph,
    params: &CutParams,
    kernels: &[Kernel],
    cone: &[bool],
) -> PyResult<CutResult> {
    if params.k < 2 || params.k > 6 {
        return Err(PyValueError::new_err(format!(
            "k must be in 2..=6, got {}",
            params.k
        )));
    }
    let t0 = Instant::now();
    let n = graph.classes.len();
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); n];

    let mut stats = CutStats {
        k: params.k,
        max_cuts: params.max_cuts,
        num_classes_graph: n,
        num_classes_cone: 0,
        num_pi_classes: 0,
        num_const_classes: 0,
        num_trivial_only: 0,
        num_cyclic_classes: 0,
        total_nontrivial_cuts: 0,
        tt_conflicts: 0,
        max_cuts_per_class: 0,
        hist: Vec::new(),
        elapsed_ms: 0,
    };
    let mut hist_map: HashMap<usize, usize> = HashMap::new();

    for (si, comp) in graph.sccs.iter().enumerate() {
        let cyclic_comp = comp.len() > 1
            || graph.adj[comp[0] as usize].iter().any(|&c| c == comp[0]);
        if cyclic_comp {
            stats.num_cyclic_classes += comp.len();
        }
        for &cv in comp {
            if cone[cv as usize] {
                stats.num_classes_cone += 1;
            }
        }

        // Cyclic SCCs (complement pairs from demorgan/double-negation rows,
        // commutativity unions, ...) are processed by FIXPOINT ITERATION like
        // the extractor's bounded relaxation: sweep members repeatedly,
        // finalizing any class whose e-nodes only reference already-finalized
        // classes, until no progress.  Members still unfinished after the loop
        // keep only their trivial cut (consumers leafify them).
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
                let kind = class_kind(graph, kernels, v)?;
                if kind != ClassKind::Internal {
                    // PI / constant: trivial cut only
                    let trivial = match kind {
                        ClassKind::Pi => {
                            stats.num_pi_classes += 1;
                            Cut { leaves: vec![cv], tt: VAR1, realization: None, nnodes: 0 }
                        }
                        ClassKind::Const(c) => {
                            stats.num_const_classes += 1;
                            Cut { leaves: vec![], tt: c, realization: None, nnodes: 0 }
                        }
                        ClassKind::Internal => unreachable!(),
                    };
                    cuts[v] = vec![trivial];
                    *hist_map.entry(1).or_insert(0) += 1;
                    stats.num_trivial_only += 1;
                    done[i] = true;
                    progress = true;
                    continue;
                }
                // usable now? (every e-node's children finalized)
                let usable = graph.enodes_of[v].iter().any(|&ei| {
                    let e = &graph.enodes[ei];
                    e.ch_eq
                        .iter()
                        .all(|&c| graph.scc_of[c as usize] as usize != si || done[scc_pos[&c]])
                });
                if !usable {
                    continue; // deferred to a later pass
                }

                // ---- trivial cut (the class itself as an opaque leaf) ----
                let trivial = Cut {
                    leaves: vec![cv],
                    tt: VAR1,
                    realization: None,
                    nnodes: 0,
                };

                // ---- merge over usable e-nodes (choice union) ----
                let mut cands: Vec<Cut> = Vec::new();
                let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
                let ens: &[usize] = match params.enode_cap {
                    Some(c) if graph.enodes_of[v].len() > c => &graph.enodes_of[v][..c],
                    _ => &graph.enodes_of[v],
                };
                for &ei in ens {
                    let e = &graph.enodes[ei];
                    // skip e-nodes with an unfinalized same-SCC child
                    if e.ch_eq.iter().any(|&c| {
                        graph.scc_of[c as usize] as usize == si && !done[scc_pos[&c]]
                    }) {
                        continue;
                    }
                    if e.ch_eq.is_empty() {
                        continue;
                    }
                    let op_kernel = match kernels[e.func] {
                        Kernel::Op { tt } => tt,
                        _ => continue,
                    };
                    let lists: Vec<&[Cut]> =
                        e.ch_eq.iter().map(|&c| cuts[c as usize].as_slice()).collect();
                    let mut budget = MERGE_BUDGET;
                    for (leaves, chosen) in merge_product(&lists, params.k, &mut budget) {
                        let child_refs: Vec<&Cut> = chosen
                            .iter()
                            .zip(e.ch_eq.iter())
                            .map(|(&ci, &c)| &cuts[c as usize][ci as usize])
                            .collect();
                        let tt = compose_tt(op_kernel, &child_refs, &leaves);
                        let nnodes = 1u16 + child_refs.iter().map(|c| c.nnodes).sum::<u16>();
                        match seen.get(&leaves) {
                            Some(&i2) => {
                                // Same leafset via a different e-node: agree
                                // only on reachable leaf values when leaves are
                                // correlated (e.g. a class and its complement
                                // both in the cut).  ABC keeps the first
                                // realization's TT; so do we.
                                if cands[i2].tt != tt {
                                    stats.tt_conflicts += 1;
                                }
                            }
                            None => {
                                seen.insert(leaves.clone(), cands.len());
                                cands.push(Cut {
                                    leaves,
                                    tt,
                                    realization: Some((ei, chosen)),
                                    nnodes,
                                });
                            }
                        }
                    }
                }

                // ---- rank, dominance-filter, cap (ABC mapperCut.c order) ----
                cands.sort_by(|a, b| {
                    a.leaves
                        .len()
                        .cmp(&b.leaves.len())
                        .then(a.nnodes.cmp(&b.nnodes))
                        .then(fnv_leaves(&a.leaves).cmp(&fnv_leaves(&b.leaves)))
                });
                cands.truncate(params.cand_cap);
                let mut kept: Vec<Cut> = Vec::with_capacity(cands.len());
                for c in cands {
                    if kept.iter().any(|k2| is_subset(&k2.leaves, &c.leaves)) {
                        continue;
                    }
                    kept.push(c);
                }
                kept.truncate(params.max_cuts.saturating_sub(1));

                let mut list = Vec::with_capacity(kept.len() + 1);
                list.push(trivial);
                list.extend(kept);
                if list.len() == 1 {
                    stats.num_trivial_only += 1;
                }
                stats.total_nontrivial_cuts += list.len() - 1;
                stats.max_cuts_per_class = stats.max_cuts_per_class.max(list.len());
                *hist_map.entry(list.len()).or_insert(0) += 1;
                cuts[v] = list;
                done[i] = true;
                progress = true;
            }
            if !progress {
                break;
            }
        }

        // members never finalized: poison -> trivial only
        for i in 0..comp.len() {
            if done[i] {
                continue;
            }
            done[i] = true;
            let cv = comp[i];
            let v = cv as usize;
            if !cone[v] {
                continue;
            }
            cuts[v] = vec![Cut { leaves: vec![cv], tt: VAR1, realization: None, nnodes: 0 }];
            *hist_map.entry(1).or_insert(0) += 1;
            stats.num_trivial_only += 1;
        }
    }

    let mut hist: Vec<(usize, usize)> = hist_map.into_iter().collect();
    hist.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    stats.hist = hist;
    stats.elapsed_ms = t0.elapsed().as_millis();
    Ok(CutResult { cuts, stats })
}

// ---------------------------------------------------------------- Python API

/// Truth table of an eggverse operator over its own inputs (parity check
/// against `eggverse.cost.operator_tt`).
#[pyfunction]
pub fn op_truth_table(op: &str) -> PyResult<u64> {
    op_tt(op).ok_or_else(|| PyValueError::new_err(format!("unknown operator {op:?}")))
}

/// Cut enumeration over the e-class graph (ABC `map`-style), operator
/// agnostic.  Read-only: produces cut sets + statistics, changes no default
/// extraction/mapping behavior.
#[pyclass(unsendable)]
pub struct CutIterator {
    pub(crate) graph: ClassGraph,
    pub(crate) kernels: Vec<Kernel>,
    pub(crate) result: CutResult,
    pub(crate) cone: Vec<bool>,
    pub(crate) root_ids: Vec<u32>,
    pub(crate) pi_names: HashMap<u32, String>,
    var_func: Option<usize>,
    const_funcs: HashMap<u64, usize>,
}

#[pymethods]
impl CutIterator {
    /// Build and run the cut enumeration.
    ///
    /// * `sort`    - name of the single eq-sort (the term sort)
    /// * `op_map`  - egg constructor name -> eggverse operator name
    ///               ("var"|"zero"|"one"|"not"|"and"|...|"mux")
    /// * `roots`   - values of the PO classes (PO cone filter; empty = whole
    ///               graph)
    /// * `k`       - max cut leaves (default 4)
    /// * `max_cuts` - max surviving cuts per class (default 16)
    /// * `enode_cap` - consider only the first N e-nodes per class
    #[new]
    #[pyo3(signature = (egraph, sort, op_map, roots, k=4, max_cuts=16, enode_cap=None, cand_cap=150))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        egraph: &EGraph,
        sort: String,
        op_map: HashMap<String, String>,
        roots: Vec<Value>,
        k: usize,
        max_cuts: usize,
        enode_cap: Option<usize>,
        cand_cap: usize,
    ) -> PyResult<Self> {
        if op_map.is_empty() {
            return Err(PyValueError::new_err("op_map must not be empty"));
        }
        let eg = &egraph.egraph;
        let heads: HashMap<String, f64> = op_map.keys().map(|n| (n.clone(), 0.0)).collect();
        let graph = ClassGraph::build(eg, &sort, &heads)?;
        let kernels = build_kernels(&graph, &op_map)?;

        let mut var_func = None;
        let mut const_funcs = HashMap::new();
        for (i, k) in kernels.iter().enumerate() {
            match k {
                Kernel::Leaf => var_func = Some(i),
                Kernel::Const(c) => {
                    const_funcs.insert(*c, i);
                }
                Kernel::Op { .. } => {}
            }
        }

        // PI class -> primitive name (for netlist emission downstream)
        let mut pi_names: HashMap<u32, String> = HashMap::new();
        for e in &graph.enodes {
            if kernels[e.func] == Kernel::Leaf {
                if let Some(&(_, pv)) = e.ch_prim.first() {
                    let name: String = eg.value_to_base::<egglog::sort::S>(pv).0;
                    pi_names.insert(e.out, name);
                }
            }
        }

        // PO cone filter
        let root_sort = eg
            .get_sort_by_name(&sort)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown sort {sort}")))?;
        let root_ids: Vec<u32> = roots
            .iter()
            .map(|v| {
                let canonical = eg.get_canonical_value(v.0, root_sort);
                graph.class_index.get(&canonical).copied().ok_or_else(|| {
                    PyValueError::new_err(
                        "root value not present in the whitelisted class graph \
                         (is its constructor in op_map?)",
                    )
                })
            })
            .collect::<PyResult<Vec<u32>>>()?;
        let cone = if root_ids.is_empty() {
            vec![true; graph.classes.len()]
        } else {
            graph.cone_mask(&root_ids)
        };

        let params = CutParams {
            k,
            max_cuts,
            enode_cap,
            cand_cap,
        };
        let result = enumerate_cuts(&graph, &params, &kernels, &cone)?;

        Ok(CutIterator {
            graph,
            kernels,
            result,
            cone,
            root_ids,
            pi_names,
            var_func,
            const_funcs,
        })
    }

    /// Enumeration statistics.
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let s = &self.result.stats;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("k", s.k)?;
        d.set_item("max_cuts", s.max_cuts)?;
        d.set_item("num_classes_graph", s.num_classes_graph)?;
        d.set_item("num_classes_cone", s.num_classes_cone)?;
        d.set_item("num_pi_classes", s.num_pi_classes)?;
        d.set_item("num_const_classes", s.num_const_classes)?;
        d.set_item("num_trivial_only", s.num_trivial_only)?;
        d.set_item("num_cyclic_classes", s.num_cyclic_classes)?;
        d.set_item("total_nontrivial_cuts", s.total_nontrivial_cuts)?;
        d.set_item("tt_conflicts", s.tt_conflicts)?;
        d.set_item("max_cuts_per_class", s.max_cuts_per_class)?;
        let hist = pyo3::types::PyDict::new(py);
        for (n, c) in s.hist.iter().take(16) {
            hist.set_item(n, c)?;
        }
        d.set_item("cuts_per_class_hist", hist)?;
        d.set_item("elapsed_ms", s.elapsed_ms)?;
        Ok(d.into_any().unbind())
    }

    /// Cut list of one class, [{"leaves": [class ids], "tt": int,
    /// "nnodes": int, "enode": egg ctor name | None, "child_cuts": [...]}].
    /// The first entry is always the trivial cut.
    #[pyo3(signature = (egraph, value, sort))]
    fn cuts_of<'py>(
        &self,
        egraph: &EGraph,
        value: Value,
        sort: String,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let idx = self.class_of_impl(&egraph.egraph, value, &sort)?;
        let out = pyo3::types::PyList::empty(py);
        for c in &self.result.cuts[idx as usize] {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("leaves", c.leaves.clone())?;
            d.set_item("tt", c.tt)?;
            d.set_item("nnodes", c.nnodes)?;
            match &c.realization {
                Some((ei, cc)) => {
                    d.set_item("enode", self.graph.funcs[self.graph.enodes[*ei].func].term_name.clone())?;
                    d.set_item("child_cuts", cc.clone())?;
                }
                None => {
                    d.set_item("enode", Option::<String>::None)?;
                    d.set_item("child_cuts", Option::<Vec<u32>>::None)?;
                }
            }
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// Reconstruct the realizing term of cut `cut_idx` of the class of
    /// `value`, with the cut leaves substituted by fresh PIs named
    /// `n0..n{k-1}` (constants stay structural).  This is the soundness-
    /// checking vehicle: simulating the term over its PIs must reproduce the
    /// cut's truth table bit for bit.
    #[pyo3(signature = (egraph, termdag, value, cut_idx, sort))]
    fn cut_term(
        &self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        cut_idx: usize,
        sort: String,
    ) -> PyResult<TermId> {
        let idx = self.class_of_impl(&egraph.egraph, value, &sort)?;
        let list = &self.result.cuts[idx as usize];
        if cut_idx >= list.len() {
            return Err(PyValueError::new_err(format!(
                "cut_idx {cut_idx} out of range (class has {} cuts)",
                list.len()
            )));
        }
        let root_leaves = list[cut_idx].leaves.clone();
        let var_name = self
            .var_func
            .map(|f| self.graph.funcs[f].term_name.clone())
            .ok_or_else(|| PyValueError::new_err("no var constructor in op_map"))?;

        enum Frame {
            Enter(u32, usize),
            Build {
                class: u32,
                cut: usize,
                slots: Vec<Option<TermId>>,
                n_eq: usize,
            },
        }
        let mut memo: HashMap<(u32, usize), TermId> = HashMap::new();
        let mut stack = vec![Frame::Enter(idx, cut_idx)];
        let mut results: Vec<TermId> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(v, ci) => {
                    if let Some(&t) = memo.get(&(v, ci)) {
                        results.push(t);
                        continue;
                    }
                    let cut = &self.result.cuts[v as usize][ci];
                    match &cut.realization {
                        None => {
                            // leaf: PI/internal class -> fresh variable,
                            // constant class -> the constant constructor
                            let t = if let Some(pos) = root_leaves.iter().position(|&l| l == v) {
                                let lit =
                                    termdag.0.lit(Literal::String(format!("n{pos}")));
                                termdag.0.app(var_name.clone(), vec![lit])
                            } else {
                                let kind = class_kind(&self.graph, &self.kernels, v as usize)?;
                                match kind {
                                    ClassKind::Const(c) => {
                                        let fid = self.const_funcs.get(&c).ok_or_else(|| {
                                            PyValueError::new_err("no constructor for constant class")
                                        })?;
                                        let name = self.graph.funcs[*fid].term_name.clone();
                                        termdag.0.app(name, Vec::new())
                                    }
                                    _ => {
                                        return Err(PyValueError::new_err(
                                            "leaf class outside root leaves and not constant",
                                        ))
                                    }
                                }
                            };
                            memo.insert((v, ci), t);
                            results.push(t);
                        }
                        Some((ei, child_cuts)) => {
                            let e = &self.graph.enodes[*ei];
                            let mask = &self.graph.funcs[e.func].eq_mask;
                            let arity = self.graph.funcs[e.func].arity;
                            let mut slots: Vec<Option<TermId>> = vec![None; arity];
                            let mut to_enter: Vec<(u32, usize)> = Vec::new();
                            let mut eq_iter = e.ch_eq.iter();
                            let mut prim_iter = e.ch_prim.iter();
                            for (pos, is_eq) in mask.iter().enumerate() {
                                if *is_eq {
                                    let Some(&c) = eq_iter.next() else {
                                        return Err(PyValueError::new_err("corrupt enode"));
                                    };
                                    let Some(&cci) = child_cuts.get(to_enter.len()) else {
                                        return Err(PyValueError::new_err("corrupt cut realization"));
                                    };
                                    to_enter.push((c, cci as usize));
                                } else {
                                    let Some(&(_, pv)) = prim_iter.next() else {
                                        return Err(PyValueError::new_err("corrupt enode"));
                                    };
                                    let s: String =
                                        egraph.egraph.value_to_base::<egglog::sort::S>(pv).0;
                                    slots[pos] = Some(termdag.0.lit(Literal::String(s)));
                                }
                            }
                            let n_eq = to_enter.len();
                            stack.push(Frame::Build {
                                class: v,
                                cut: ci,
                                slots,
                                n_eq,
                            });
                            for &(c, cci) in to_enter.iter().rev() {
                                stack.push(Frame::Enter(c, cci));
                            }
                        }
                    }
                }
                Frame::Build {
                    class,
                    cut,
                    mut slots,
                    n_eq,
                } => {
                    let mut popped: Vec<TermId> = Vec::with_capacity(n_eq);
                    for _ in 0..n_eq {
                        popped.push(
                            results
                                .pop()
                                .ok_or_else(|| PyValueError::new_err("corrupt reconstruction"))?,
                        );
                    }
                    popped.reverse();
                    let mut it = popped.into_iter();
                    for s in slots.iter_mut() {
                        if s.is_none() {
                            *s = Some(
                                it.next()
                                    .ok_or_else(|| PyValueError::new_err("corrupt reconstruction"))?,
                            );
                        }
                    }
                    let children: Vec<TermId> = slots
                        .into_iter()
                        .map(|s| s.expect("all slots filled"))
                        .collect();
                    let name = {
                        let cut_ref = &self.result.cuts[class as usize][cut];
                        let (rfun, _) = cut_ref
                            .realization
                            .as_ref()
                            .expect("Build frame implies realization");
                        self.graph.funcs[self.graph.enodes[*rfun].func].term_name.clone()
                    };
                    let t = termdag.0.app(name, children);
                    memo.insert((class, cut), t);
                    results.push(t);
                }
            }
        }
        results
            .pop()
            .ok_or_else(|| PyValueError::new_err("empty reconstruction"))
    }

    /// Total number of enumerated classes (debug/introspection).
    #[getter]
    fn num_classes(&self) -> usize {
        self.graph.classes.len()
    }

    /// Internal class id of `value` (must be of the root sort).
    #[pyo3(signature = (egraph, value, sort))]
    fn class_of(&self, egraph: &EGraph, value: Value, sort: String) -> PyResult<u32> {
        self.class_of_impl(&egraph.egraph, value, &sort)
    }

    /// E-nodes of one class by internal id (debug/diagnosis).
    fn enodes_of_class<'py>(&self, class: usize, py: Python<'py>) -> PyResult<PyObject> {
        let out = pyo3::types::PyList::empty(py);
        let Some(eis) = self.graph.enodes_of.get(class) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(class));
        };
        match class_kind(&self.graph, &self.kernels, class) {
            Ok(crate::cut_iter::ClassKind::Pi) => {
                let _ = 0;
            }
            _ => {}
        }
        for &ei in eis {
            let e = &self.graph.enodes[ei];
            let d = pyo3::types::PyDict::new(py);
            d.set_item("func", self.graph.funcs[e.func].term_name.clone())?;
            d.set_item("ch_eq", e.ch_eq.clone())?;
            d.set_item("scc", self.graph.scc_of.get(class).copied())?;
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }

    /// Cut list of one class by internal id (debug/diagnosis; same shape as
    /// `cuts_of`).
    fn cuts_of_class<'py>(&self, class: usize, py: Python<'py>) -> PyResult<PyObject> {
        let out = pyo3::types::PyList::empty(py);
        let Some(list) = self.result.cuts.get(class) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(class));
        };
        for c in list {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("leaves", c.leaves.clone())?;
            d.set_item("tt", c.tt)?;
            d.set_item("nnodes", c.nnodes)?;
            match &c.realization {
                Some((ei, cc)) => {
                    d.set_item("enode", self.graph.funcs[self.graph.enodes[*ei].func].term_name.clone())?;
                    d.set_item("child_cuts", cc.clone())?;
                }
                None => {
                    d.set_item("enode", Option::<String>::None)?;
                    d.set_item("child_cuts", Option::<Vec<u32>>::None)?;
                }
            }
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }
}

impl CutIterator {
    pub(crate) fn class_of_impl(&self, eg: &egglog::EGraph, value: Value, sort: &str) -> PyResult<u32> {
        let root = eg
            .get_sort_by_name(sort)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown sort {sort}")))?;
        let canonical = eg.get_canonical_value(value.0, root);
        self.graph.class_index.get(&canonical).copied().ok_or_else(|| {
            PyValueError::new_err("Unextractable root: value not in the whitelisted class graph")
        })
    }
}

// -------------------------------------------------------------------- tests
// Hand-computed fixtures over synthetic class graphs (no egglog involved).
// Class ids are the graph's internal indices; tt bit m = value at the
// minterm with pin i = bit i of m.

#[cfg(test)]
pub(crate) mod fixtures {
    use super::{CutParams, CutResult};
    use crate::class_graph::{ClassGraph, Enode, FuncTab, tarjan_scc};
    use egglog::Value as EggValue;
    use std::collections::HashMap;

    /// n_classes explicit; (term_name, eq_mask) funcs; enodes as (func, out, ch_eq).
    pub(crate) fn mkgraph(
        n_classes: usize,
        funcs: Vec<(&str, Vec<bool>)>,
        ens: Vec<(usize, usize, Vec<u32>)>,
    ) -> ClassGraph {
        let mut g = ClassGraph {
            funcs: funcs
                .into_iter()
                .map(|(n, m)| {
                    let arity = m.len();
                    FuncTab {
                        term_name: n.to_string(),
                        eq_mask: m,
                        arity,
                    }
                })
                .collect(),
            enodes: Vec::new(),
            classes: Vec::new(),
            class_index: HashMap::new(),
            enodes_of: Vec::new(),
            adj: Vec::new(),
            sccs: Vec::new(),
            scc_of: Vec::new(),
        };
        let mut ens_of: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
        for (func, out, ch) in &ens {
            assert!((*out as usize) < n_classes);
            for &c in ch {
                assert!((c as usize) < n_classes);
            }
            ens_of[*out as usize].push(g.enodes.len());
            g.enodes.push(Enode {
                func: *func,
                out: *out as u32,
                ch_eq: ch.clone(),
                ch_prim: Vec::new(),
                head: 0.0,
            });
        }
        g.classes = (0..n_classes as u32).map(EggValue::new_const).collect();
        g.enodes_of = ens_of;
        g.adj = vec![Vec::new(); n_classes];
        for e in &g.enodes {
            for &c in &e.ch_eq {
                g.adj[e.out as usize].push(c);
            }
        }
        g.sccs = tarjan_scc(n_classes, &g.adj);
        g.scc_of = vec![0u32; n_classes];
        for (si, comp) in g.sccs.iter().enumerate() {
            for &v in comp {
                g.scc_of[v as usize] = si as u32;
            }
        }
        g
    }

    pub(crate) fn params(k: usize, max_cuts: usize) -> CutParams {
        CutParams {
            k,
            max_cuts,
            enode_cap: None,
            cand_cap: 1000,
        }
    }

    pub(crate) fn var_mask_pub(i: usize, n: usize) -> u64 {
        super::var_mask(i, n)
    }

    pub(crate) fn leaf_leaves(res: &CutResult, v: usize) -> Vec<Vec<u32>> {
        res.cuts[v].iter().map(|c| c.leaves.clone()).collect()
    }

}

#[cfg(test)]
mod tests {
    use super::fixtures::{leaf_leaves, mkgraph, params};
    use super::*;
    use crate::class_graph::Enode;
    use egglog::Value as EggValue;

    /// T1: a single PI class gets exactly its trivial cut (tt = the variable).
    #[test]
    fn t1_single_pi_trivial_only() {
        let mut g = mkgraph(1, vec![("var", vec![false])], vec![]);
        // the var e-node itself (String prim child; mkgraph only does eq children)
        g.enodes.push(Enode {
            func: 0,
            out: 0,
            ch_eq: vec![],
            ch_prim: vec![(0, EggValue::new_const(7))],
            head: 0.0,
        });
        g.enodes_of[0].push(0);
        let kernels = vec![Kernel::Leaf];
        let cone = vec![true];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(res.cuts[0].len(), 1);
        assert_eq!(res.cuts[0][0].leaves, vec![0]);
        assert_eq!(res.cuts[0][0].tt, VAR1);
        assert!(res.cuts[0][0].realization.is_none());
        assert_eq!(res.stats.num_pi_classes, 1);
    }

    /// T2: one and-node: cuts = [trivial, {x,y}] with tt = 0b1000.
    #[test]
    fn t2_single_and() {
        // classes: 0=x(var), 1=y(var), 2=and(x,y)
        let g = mkgraph(3, vec![("var", vec![false]), ("B_and", vec![true, true])], vec![(1, 2, vec![0, 1])]);
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("and").unwrap() }];
        let cone = vec![true, true, true];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(leaf_leaves(&res, 2), vec![vec![2], vec![0, 1]]);
        assert_eq!(res.cuts[2][1].tt, 0b1000);
    }

    /// T3: shared child: w = and(x, z), z = and(x, y).  w must see both
    /// {x,y} (through z's non-trivial cut) and {x,z} (through z's trivial).
    #[test]
    fn t3_shared_child() {
        // 0=x, 1=y, 2=z=and(x,y), 3=w=and(x,z)
        let g = mkgraph(
            4,
            vec![("var", vec![false]), ("B_and", vec![true, true])],
            vec![(1, 2, vec![0, 1]), (1, 3, vec![0, 2])],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("and").unwrap() }];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        let w = leaf_leaves(&res, 3);
        assert!(w.contains(&vec![0, 1]), "w cuts must contain x,y: {w:?}");
        assert!(w.contains(&vec![0, 2]), "w cuts must contain x,z: {w:?}");
        // tt of {x,y} in w: and(x, and(x,y)) = x*y = 0b1000
        let i = w.iter().position(|l| l == &vec![0, 1]).unwrap();
        assert_eq!(res.cuts[3][i].tt, 0b1000);
    }

    /// T4: choice class with two enodes over the same leafset (and(x,y) and
    /// and(y,x)) dedupes to ONE {x,y} cut; tt consistent across realizations.
    #[test]
    fn t4_choice_dedupe() {
        // 0=x, 1=y, 2=choice{and(x,y), and(y,x)}
        let g = mkgraph(
            3,
            vec![("var", vec![false]), ("B_and", vec![true, true])],
            vec![(1, 2, vec![0, 1]), (1, 2, vec![1, 0])],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("and").unwrap() }];
        let cone = vec![true; 3];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(leaf_leaves(&res, 2), vec![vec![2], vec![0, 1]]);
    }

    /// T4b: constant class: leaves = [], tt = the constant bit; consumers
    /// constant-fold through it.
    #[test]
    fn t4b_const_class() {
        // 0=x, 1=zero, 2=and(x, zero)
        let g = mkgraph(
            3,
            vec![("var", vec![false]), ("B_zero", vec![]), ("B_and", vec![true, true])],
            vec![(1, 1, vec![]), (2, 2, vec![0, 1])],
        );
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Const(0),
            Kernel::Op { tt: op_tt("and").unwrap() },
        ];
        let cone = vec![true; 3];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(res.cuts[1].len(), 1);
        assert_eq!(res.cuts[1][0].leaves, Vec::<u32>::new());
        assert_eq!(res.cuts[1][0].tt, 0);
        // and(x, 0) = 0
        assert_eq!(leaf_leaves(&res, 2), vec![vec![2], vec![0]]);
        assert_eq!(res.cuts[2][1].tt, 0);
    }

    /// T5: 3-level chain: cut-set growth at K=3 vs K=4 (hand-computed).
    #[test]
    fn t5_chain_growth() {
        // 0=a, 1=b, 2=c, 3=d, 4=and(a,b), 5=and(4,c), 6=and(5,d)
        let g = mkgraph(
            7,
            vec![("var", vec![false]), ("B_and", vec![true, true])],
            vec![
                (1, 4, vec![0, 1]),
                (1, 5, vec![4, 2]),
                (1, 6, vec![5, 3]),
            ],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("and").unwrap() }];
        let cone = vec![true; 7];
        let res3 = enumerate_cuts(&g, &params(3, 16), &kernels, &cone).unwrap();
        // class 4: {4}, {a,b}
        assert_eq!(leaf_leaves(&res3, 4), vec![vec![4], vec![0, 1]]);
        // class 5: {5}, {4,c}, {a,b,c}
        assert_eq!(leaf_leaves(&res3, 5), vec![vec![5], vec![2, 4], vec![0, 1, 2]]);
        // class 6 at K=3: from {5},{4,c},{a,b,c} x {d}: {6}, {5,d}, {4,c,d}, {a,b,c,d}-
        //   the last exceeds K=3 -> dropped
        assert_eq!(leaf_leaves(&res3, 6), vec![vec![6], vec![3, 5], vec![2, 3, 4]]);
        let res4 = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        // class 6 at K=4 adds {a,b,c,d}
        let w = leaf_leaves(&res4, 6);
        assert!(w.contains(&vec![0, 1, 2, 3]), "K=4 must admit the 4-leaf cut: {w:?}");
        // tt of the full cut = and(and(and(a,b),c),d) = 0b1000_0000_0000_0000
        let i = w.iter().position(|l| l == &vec![0, 1, 2, 3]).unwrap();
        assert_eq!(res4.cuts[6][i].tt, 1 << 15);
    }

    /// T6: mux spine: 3-ary merge via 2-level fold produces the {s,a,b} cut
    /// with the mux truth table.
    #[test]
    fn t6_mux() {
        // 0=s, 1=a, 2=b, 3=mux(s,a,b)
        let g = mkgraph(
            4,
            vec![("var", vec![false]), ("B_mux", vec![true, true, true])],
            vec![(1, 3, vec![0, 1, 2])],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("mux").unwrap() }];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(leaf_leaves(&res, 3), vec![vec![3], vec![0, 1, 2]]);
        let tt = res.cuts[3][1].tt;
        // spot-check mux semantics: minterm m, s=bit0, a=bit1, b=bit2
        for m in 0..8u64 {
            let (s, a, b) = ((m & 1) == 1, (m >> 1) & 1 == 1, (m >> 2) & 1 == 1);
            assert_eq!((tt >> m) & 1, u64::from(if s { a } else { b }), "minterm {m}");
        }
    }

    /// T7: cyclic SCC: a self-loop class enumerates to trivial-only (no
    /// hang), and downstream classes may still use it as a leaf.
    #[test]
    fn t7_self_loop_degrades_to_leaf() {
        // 0 = or(x, 0) self-loop; 1 = and(0, y); 2 = y(var); 3 = x(var)
        let g = mkgraph(
            4,
            vec![
                ("var", vec![false]),
                ("B_or", vec![true, true]),
                ("B_and", vec![true, true]),
            ],
            vec![(1, 0, vec![3, 0]), (2, 1, vec![0, 2])],
        );
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt("or").unwrap() },
            Kernel::Op { tt: op_tt("and").unwrap() },
        ];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(res.cuts[0].len(), 1, "self-loop class: trivial only");
        assert_eq!(res.stats.num_cyclic_classes, 1);
        // class 1 uses class 0 as a leaf: cuts {1}, {0,2}
        assert_eq!(leaf_leaves(&res, 1), vec![vec![1], vec![0, 2]]);
    }

    /// T8: caps (enode_cap, max_cuts) + determinism of two runs.
    #[test]
    fn t8_caps_and_determinism() {
        // wide class: 10 = or(var i, var (i+1)%10) with 10 enodes
        let funcs = vec![("var", vec![false]), ("B_or", vec![true, true])];
        let mut enodes = Vec::new();
        for i in 0..10u32 {
            let a = i as usize;
            let b = ((i + 1) % 10) as usize;
            enodes.push((1usize, 10usize, vec![a as u32, b as u32]));
        }
        let g = mkgraph(11, funcs, enodes);
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("or").unwrap() }];
        let cone = vec![true; 11];
        let mut p = params(4, 16);
        p.enode_cap = Some(2);
        let res = enumerate_cuts(&g, &p, &kernels, &cone).unwrap();
        // only enodes (0,1) and (1,2) considered -> leaf pairs {0,1},{1,2}
        let w = leaf_leaves(&res, 10);
        assert!(w.contains(&vec![0, 1]));
        assert!(w.contains(&vec![1, 2]));
        assert!(!w.contains(&vec![4, 5]));
        // max_cuts cap
        let mut p2 = params(4, 3);
        p2.enode_cap = None;
        let res2 = enumerate_cuts(&g, &p2, &kernels, &cone).unwrap();
        assert!(res2.cuts[10].len() <= 3);
        // determinism
        let res3 = enumerate_cuts(&g, &p, &kernels, &cone).unwrap();
        assert_eq!(format!("{:?}", res.cuts[10]), format!("{:?}", res3.cuts[10]));
    }

    /// T9: aoi21 kernel truth table over {x,y,z} = !((x&y)|z) = 0b0111.
    #[test]
    fn t9_aoi21() {
        // 0=x,1=y,2=z, 3=aoi21(x,y,z)
        let g = mkgraph(
            4,
            vec![("var", vec![false]), ("B_aoi21", vec![true, true, true])],
            vec![(1, 3, vec![0, 1, 2])],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("aoi21").unwrap() }];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(res.cuts[3][1].tt, 0b0111);
    }

    /// T10: explicit not-node: n = not(x) has the {x} cut with tt = 0b01.
    #[test]
    fn t10_not_node() {
        // 0=x, 1=not(x), 2=and(1, y), 3=y
        let g = mkgraph(
            4,
            vec![("var", vec![false]), ("B_not", vec![true]), ("B_and", vec![true, true])],
            vec![(1, 1, vec![0]), (2, 2, vec![1, 3])],
        );
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt("not").unwrap() },
            Kernel::Op { tt: op_tt("and").unwrap() },
        ];
        let cone = vec![true; 4];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(leaf_leaves(&res, 1), vec![vec![1], vec![0]]);
        assert_eq!(res.cuts[1][1].tt, 0b01);
        // and(~x, y): minterms where x=0,y=1 -> m = 0b10 = 2
        let w = leaf_leaves(&res, 2);
        let i = w.iter().position(|l| l == &vec![0, 3]).unwrap();
        assert_eq!(res.cuts[2][i].tt, 0b0100);
    }

    /// T19: complement pair (c, ~c) forms a 2-cycle via the double-negation
    /// rows; fixpoint iteration must finalize BOTH classes (~c maps as a
    /// not-gate over c) instead of leafifying them.
    #[test]
    fn t19_complement_pair_fixpoint() {
        // 0 = x (var row + not(1) row from double-negation), 1 = not(0)
        let mut g = mkgraph(
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
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("not").unwrap() }];
        let cone = vec![true; 2];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        // class 0: trivial only (var); class 1: trivial + {0} with tt = not
        assert_eq!(leaf_leaves(&res, 0), vec![vec![0]]);
        assert_eq!(leaf_leaves(&res, 1), vec![vec![1], vec![0]]);
        assert_eq!(res.cuts[1][1].tt, 0b01);
        assert_eq!(res.stats.num_trivial_only, 1);
    }

    /// T11: dominance — a superset-leafset cut is dropped when a subset one
    /// survives (ABC Map_CutFilter).  The choice class 4 carries a direct
    /// and(a,b) e-node (cut {a,b}) next to the nested and(and(a,b),c) shape
    /// (cut {a,b,c}); the superset must go.  (Leafset nesting, not global
    /// e-graph soundness, is what this test exercises.)
    #[test]
    fn t11_dominance() {
        // 0=a,1=b,2=c, 3=and(a,b), 4=choice{and(3,c), and(a,b)}
        let g = mkgraph(
            5,
            vec![("var", vec![false]), ("B_and", vec![true, true])],
            vec![(1, 3, vec![0, 1]), (1, 4, vec![3, 2]), (1, 4, vec![0, 1])],
        );
        let kernels = vec![Kernel::Leaf, Kernel::Op { tt: op_tt("and").unwrap() }];
        let cone = vec![true; 5];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        // {0,1,2} dropped by dominance ({0,1} subset); {2,3} incomparable, kept;
        // {2,3} ranks before {0,1} by the deterministic FNV leaf tie-break.
        assert_eq!(leaf_leaves(&res, 4), vec![vec![4], vec![2, 3], vec![0, 1]]);
    }

    /// T12: same leafset realized by two e-nodes with different free-variable
    /// TTs (possible when leaves are correlated — leaf don't-cares).  The
    /// first realization's TT is kept and the conflict is counted, matching
    /// ABC's dedupe-by-leafset behavior.
    #[test]
    fn t12_tt_conflict_keeps_first() {
        // class 2 carries and(x,y) and or(x,y) over the same leaves {x,y}:
        // unreachable-assignment divergence, first (and) TT kept.
        let g = mkgraph(
            3,
            vec![("var", vec![false]), ("B_and", vec![true, true]), ("B_or", vec![true, true])],
            vec![(1, 2, vec![0, 1]), (2, 2, vec![0, 1])],
        );
        let kernels = vec![
            Kernel::Leaf,
            Kernel::Op { tt: op_tt("and").unwrap() },
            Kernel::Op { tt: op_tt("or").unwrap() },
        ];
        let cone = vec![true; 3];
        let res = enumerate_cuts(&g, &params(4, 16), &kernels, &cone).unwrap();
        assert_eq!(leaf_leaves(&res, 2), vec![vec![2], vec![0, 1]]);
        assert_eq!(res.cuts[2][1].tt, 0b1000); // the FIRST realization (and)
        assert_eq!(res.stats.tt_conflicts, 1);
    }
}
