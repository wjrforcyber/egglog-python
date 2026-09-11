//! DAG-specialized extractor: Tarjan SCC + levelized dynamic programming.
//!
//! Motivation (eggverse): egglog's default extractor is a Bellman-Ford style
//! relaxation over ALL constructor tables until global fixpoint, and with a
//! Python cost model every relaxation crosses into Python.  For synthesis
//! e-graphs (positive, monotone costs; sparse cycles from commutativity /
//! absorption unions) the same optimal per-class costs can be computed with:
//!
//! 1. condense the e-class digraph into SCCs (iterative Tarjan, O(V+E));
//! 2. sweep SCCs in topological order (Tarjan emits children-first);
//! 3. one-shot min for acyclic singleton SCCs; tiny bounded relaxation for
//!    cyclic SCCs (positive costs converge immediately);
//! 4. deterministic structural tie-break noise (stable across runs), which
//!    removes the arrival-order dependence that can leave the default
//!    extractor without a cycle-free parent edge (its `unwrap` panic).
//!
//! Costs are provided as a pure-Rust table (constructor egg-name -> head
//! cost) plus a fold kind, so relaxation never calls into Python.
//!
//! The class-graph snapshot (whitelisted constructor enumeration, union-find
//! canonicalization, class interning, SCC condensation) lives in
//! `class_graph.rs` and is shared with the cut iterator.
//!
//! Scope: single eq-sort root (the caller's term sort), constructor
//! whitelist vocabulary, primitives limited to `String` children.  The
//! original `Extractor` remains the general-purpose fallback.

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::class_graph::{ClassGraph, SccProfile};
use crate::{egraph::EGraph, egraph::Value, termdag::TermDag};
use egglog::TermId;
use egglog::ast::Literal;

const FNV_OFF: u64 = 0xcbf29ce484222325;

fn fnv_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv_mix_u64(mut h: u64, w: u64) -> u64 {
    for b in w.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[pyclass(unsendable)]
pub struct DagExtractor {
    graph: ClassGraph,
    costs: Vec<f64>,
    best: Vec<Option<usize>>,
    /// Persistent reconstruction memo (class idx -> TermId), shared by all
    /// `extract_best` calls on this extractor (the TermDag is shared too).
    memo: HashMap<u32, TermId>,
    scc_profile: Option<SccProfile>,
}

#[pymethods]
impl DagExtractor {
    /// Build the extractor, running the SCC condensation + levelized DP.
    ///
    /// * `sort`        - name of the single eq-sort to extract from
    /// * `head_costs`  - egg constructor name -> head cost (f64 >= 0);
    ///                    constructors not listed are ignored (whitelist)
    /// * `fold`        - "sum": head + sum(children)
    ///                    "max": head + max(positive children | 0) + eps*sum(children)
    /// * `noise_scale`   - scale of the deterministic tie-break noise
    /// * `profile_sccs`  - when true, compute (and expose via the
    ///                      `scc_profile` getter) statistics about the SCC
    ///                      condensation: node coverage by component kind,
    ///                      size mean/variance/median/max and a size
    ///                      histogram.  Purely observational: the DP, costs
    ///                      and extracted terms are identical whether or not
    ///                      this is enabled (default false).
    ///
    /// Mirrors eggverse's Python cost models: a head cost of exactly 0 makes
    /// the enode cost 0 regardless of children (constants / variables).
    #[new]
    #[pyo3(signature = (egraph, sort, head_costs, fold="sum", eps=0.0, noise_scale=1e-7, profile_sccs=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        egraph: &EGraph,
        sort: String,
        head_costs: HashMap<String, f64>,
        fold: &str,
        eps: f64,
        noise_scale: f64,
        profile_sccs: bool,
    ) -> PyResult<Self> {
        let fold_max = match fold {
            "sum" => false,
            "max" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown fold {other:?}; expected \"sum\" or \"max\""
                )))
            }
        };
        let graph = ClassGraph::build(&egraph.egraph, &sort, &head_costs)?;
        let n = graph.classes.len();

        // ---- levelized DP over SCCs (children-first order) ----
        let mut costs: Vec<f64> = vec![f64::INFINITY; n];
        let mut best: Vec<Option<usize>> = vec![None; n];
        // E4-determinism: bottom-up STRUCTURAL hash per class.  Tie-breaks use
        // this instead of raw Value bits / enode ids, so equal-cost picks are
        // stable even if e-class numbering or intern order shifts per process.
        let mut sh: Vec<u64> = vec![0; n];

        let enode_struct_sh =
            |e: &crate::class_graph::Enode, sh: &Vec<u64>| -> u64 {
                let mut h = fnv_bytes(FNV_OFF, graph.funcs[e.func].term_name.as_bytes());
                for &c in &e.ch_eq {
                    h = fnv_mix_u64(h, sh[c as usize]);
                }
                for (pos, v) in &e.ch_prim {
                    h = fnv_mix_u64(h, *pos as u64);
                    let mut hs = std::collections::hash_map::DefaultHasher::new();
                    use std::hash::{Hash, Hasher};
                    v.hash(&mut hs);
                    h = fnv_mix_u64(h, hs.finish());
                }
                h
            };

        let enode_cost =
            |e: &crate::class_graph::Enode, costs: &Vec<f64>, sh: &Vec<u64>| -> Option<f64> {
                if e.head == 0.0 {
                    return Some(0.0);
                }
                let mut total = 0.0;
                let mut max_pos = 0.0f64;
                for &c in &e.ch_eq {
                    let cc = costs[c as usize];
                    if !cc.is_finite() {
                        return None; // child unfinalized (same SCC) or unextractable
                    }
                    if cc > max_pos {
                        max_pos = cc;
                    }
                    total += cc;
                }
                // structural noise (was: hash over raw child Value bits)
                let noise = (enode_struct_sh(e, sh) % 1024) as f64 * noise_scale;
                if fold_max {
                    Some(e.head + max_pos + eps * total + noise)
                } else {
                    Some(e.head + total + noise)
                }
            };

        for comp in &graph.sccs {
            let has_self_loop = comp.len() > 1
                || graph.adj[comp[0] as usize]
                    .iter()
                    .any(|&c| c == comp[0]);
            if !has_self_loop {
                let v = comp[0] as usize;
                let mut bc = f64::INFINITY;
                let mut bsh = u64::MAX;
                let mut bi: Option<usize> = None;
                for &ei in &graph.enodes_of[v] {
                    if let Some(c) = enode_cost(&graph.enodes[ei], &costs, &sh) {
                        let esh = enode_struct_sh(&graph.enodes[ei], &sh);
                        if c < bc || (c == bc && esh < bsh) {
                            bc = c;
                            bsh = esh;
                            bi = Some(ei);
                        }
                    }
                }
                costs[v] = bc;
                sh[v] = bsh;
                best[v] = bi;
            } else {
                let members: Vec<(u32, Vec<usize>)> = comp
                    .iter()
                    .map(|v| (*v, graph.enodes_of[*v as usize].clone()))
                    .collect();
                let mut changed = true;
                let mut iters = 0usize;
                let cap = comp.len() * comp.len() + 1000;
                while changed {
                    changed = false;
                    iters += 1;
                    if iters > cap {
                        return Err(PyValueError::new_err(
                            "SCC relaxation did not converge (cost model not monotone?)",
                        ));
                    }
                    for (v, ens) in &members {
                        let vi = *v as usize;
                        for &ei in ens {
                            if let Some(c) = enode_cost(&graph.enodes[ei], &costs, &sh) {
                                let esh = enode_struct_sh(&graph.enodes[ei], &sh);
                                let better = match best[vi] {
                                    None => true,
                                    Some(cur) => {
                                        c < costs[vi]
                                            || (c == costs[vi]
                                                && esh < enode_struct_sh(&graph.enodes[cur], &sh))
                                    }
                                };
                                if better {
                                    costs[vi] = c;
                                    sh[vi] = esh;
                                    best[vi] = Some(ei);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---- optional determinism diagnostic ----
        if std::env::var_os("EGGLOG_EXTRACT_DEBUG").is_some() {
            let mut cost_sum: u64 = 0;
            let mut struct_fnv: u64 = FNV_OFF;
            let mut ids_fnv: u64 = FNV_OFF;
            for (i, b) in best.iter().enumerate() {
                if let Some(ei) = b {
                    cost_sum = cost_sum.wrapping_add(costs[i].to_bits());
                    struct_fnv = fnv_mix_u64(struct_fnv, sh[i]);
                    ids_fnv = fnv_mix_u64(ids_fnv, *ei as u64);
                }
            }
            eprintln!(
                "[xdbg] classes={cls} cost_sum={cost_sum:#x} struct_fnv={struct_fnv:#x} ids_fnv={ids_fnv:#x}",
                cls = graph.classes.len()
            );
        }

        // ---- optional SCC profiling (purely observational) ----
        let scc_profile = if profile_sccs {
            Some(SccProfile::compute(&graph.sccs, &graph.adj))
        } else {
            None
        };

        Ok(DagExtractor {
            graph,
            costs,
            best,
            memo: HashMap::default(),
            scc_profile,
        })
    }

    /// Extract the minimum-cost term for `value` (must be of the root sort).
    /// Reuses the DP results and a reconstruction memo shared across calls.
    #[pyo3(signature = (egraph, termdag, value, sort))]
    fn extract_best(
        &mut self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: String,
    ) -> PyResult<(f64, TermId)> {
        let eg = &egraph.egraph;
        let root = eg
            .get_sort_by_name(&sort)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown sort {sort}")))?;
        let canonical = eg.get_canonical_value(value.0, root);
        let Some(&idx) = self.graph.class_index.get(&canonical) else {
            return Err(PyValueError::new_err("Unextractable root"));
        };
        let cost = self.costs[idx as usize];
        if !cost.is_finite() {
            return Err(PyValueError::new_err("Unextractable root"));
        }
        let term = self.build_term(eg, termdag, idx)?;
        Ok((cost, term))
    }

    /// Number of DP classes (debug/introspection).
    #[getter]
    fn num_classes(&self) -> usize {
        self.graph.classes.len()
    }

    /// Number of whitelisted enodes (debug/introspection).
    #[getter]
    fn num_enodes(&self) -> usize {
        self.graph.enodes.len()
    }

    /// SCC condensation statistics, or None when the extractor was built
    /// without `profile_sccs=True`.
    #[getter]
    fn scc_profile<'py>(&self, py: Python<'py>) -> Option<PyObject> {
        self.scc_profile.as_ref().map(|p| p.to_py_dict(py))
    }
}

impl DagExtractor {
    /// Iterative post-order reconstruction of the minimum term for `idx`,
    /// memoized across calls.  Each completed subtree pushes exactly one
    /// TermId onto `results`; a `Build` frame pops exactly its eq-child
    /// results (primitives are filled immediately into their slots).
    fn build_term(
        &mut self,
        eg: &egglog::EGraph,
        termdag: &mut TermDag,
        idx: u32,
    ) -> PyResult<TermId> {
        if let Some(&t) = self.memo.get(&idx) {
            return Ok(t);
        }
        enum Frame {
            Enter(u32),
            Build {
                class: u32,
                slots: Vec<Option<TermId>>,
                n_eq: usize,
            },
        }
        let mut stack = vec![Frame::Enter(idx)];
        let mut results: Vec<TermId> = Vec::new();

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(v) => {
                    if let Some(&t) = self.memo.get(&v) {
                        results.push(t);
                        continue;
                    }
                    let Some(ei) = self.best[v as usize] else {
                        return Err(PyValueError::new_err("Unextractable class"));
                    };
                    let arity = self.graph.funcs[self.graph.enodes[ei].func].arity;
                    if arity == 0 {
                        let name = self.graph.funcs[self.graph.enodes[ei].func].term_name.clone();
                        let t = termdag.0.app(name, Vec::new());
                        self.memo.insert(v, t);
                        results.push(t);
                        continue;
                    }
                    let mut slots: Vec<Option<TermId>> = vec![None; arity];
                    let mut to_enter: Vec<u32> = Vec::new();
                    {
                        let e = &self.graph.enodes[ei];
                        let mask = &self.graph.funcs[e.func].eq_mask;
                        let mut eq_iter = e.ch_eq.iter();
                        let mut prim_iter = e.ch_prim.iter();
                        for (pos, is_eq) in mask.iter().enumerate() {
                            if *is_eq {
                                let Some(&c) = eq_iter.next() else {
                                    return Err(PyValueError::new_err("corrupt enode"));
                                };
                                to_enter.push(c);
                            } else {
                                let Some(&(_, pv)) = prim_iter.next() else {
                                    return Err(PyValueError::new_err("corrupt enode"));
                                };
                                let s: String = eg.value_to_base::<egglog::sort::S>(pv).0;
                                slots[pos] = Some(termdag.0.lit(Literal::String(s)));
                            }
                        }
                    }
                    let n_eq = to_enter.len();
                    stack.push(Frame::Build {
                        class: v,
                        slots,
                        n_eq,
                    });
                    // push eq children in reverse so the leftmost completes
                    // first (matches the pop-and-reverse below)
                    for c in to_enter.into_iter().rev() {
                        stack.push(Frame::Enter(c));
                    }
                }
                Frame::Build {
                    class,
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
                    let ei = self.best[class as usize].expect("best edge exists");
                    let name = self.graph.funcs[self.graph.enodes[ei].func].term_name.clone();
                    let t = termdag.0.app(name, children);
                    self.memo.insert(class, t);
                    results.push(t);
                }
            }
        }
        results
            .pop()
            .ok_or_else(|| PyValueError::new_err("empty reconstruction"))
    }
}
