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
//! Scope: single eq-sort root (the caller's term sort), constructor
//! whitelist vocabulary, primitives limited to `String` children.  The
//! original `Extractor` remains the general-purpose fallback.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{egraph::EGraph, egraph::Value, termdag::TermDag};
use egglog::ast::Literal;
use egglog::TermId;
use egglog::Value as EggValue;

const UNSUPPORTED: &str = "DagExtractor supports a single eq-sort root, whitelisted constructors, and String primitives only";

struct FuncTab {
    /// Term head used for reconstruction (the egg function name).
    term_name: String,
    /// Per child position: true if the child sort is the eq sort (a class).
    eq_mask: Vec<bool>,
    arity: usize,
}

struct Enode {
    func: usize,
    /// Class index of the output.
    out: u32,
    /// Class indices of eq children (in eq-position order).
    ch_eq: Vec<u32>,
    /// (position, value) of primitive (String) children.
    ch_prim: Vec<(u32, EggValue)>,
    /// All child values, for the deterministic tie-break hash.
    all_children: Vec<EggValue>,
    head: f64,
}

#[pyclass(unsendable)]
pub struct DagExtractor {
    funcs: Vec<FuncTab>,
    enodes: Vec<Enode>,
    classes: Vec<EggValue>,
    costs: Vec<f64>,
    best: Vec<Option<usize>>,
    /// Persistent reconstruction memo (class idx -> TermId), shared by all
    /// `extract_best` calls on this extractor (the TermDag is shared too).
    memo: HashMap<u32, TermId>,
}

fn fnv_hash(bytes: &[u8], vals: &[EggValue]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for v in vals {
        let mut hs = std::collections::hash_map::DefaultHasher::new();
        v.hash(&mut hs);
        for b in hs.finish().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
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
    /// * `noise_scale` - scale of the deterministic tie-break noise
    ///
    /// Mirrors eggverse's Python cost models: a head cost of exactly 0 makes
    /// the enode cost 0 regardless of children (constants / variables).
    #[new]
    #[pyo3(signature = (egraph, sort, head_costs, fold="sum", eps=0.0, noise_scale=1e-7))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        egraph: &EGraph,
        sort: String,
        head_costs: HashMap<String, f64>,
        fold: &str,
        eps: f64,
        noise_scale: f64,
    ) -> PyResult<Self> {
        let eg = &egraph.egraph;
        let root = eg
            .get_sort_by_name(&sort)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown sort {sort}")))?
            .clone();
        if !root.is_eq_sort() {
            return Err(PyValueError::new_err(format!("Root sort {sort} is not an eq sort")));
        }
        let fold_max = match fold {
            "sum" => false,
            "max" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown fold {other:?}; expected \"sum\" or \"max\""
                )))
            }
        };

        let mut funcs: Vec<FuncTab> = Vec::new();
        let mut enodes: Vec<Enode> = Vec::new();
        let mut class_index: HashMap<EggValue, u32> = HashMap::default();
        let mut classes: Vec<EggValue> = Vec::new();
        let mut enodes_of: Vec<Vec<usize>> = Vec::new();

        for (name, head) in &head_costs {
            let Some(f) = eg.get_function(name) else {
                continue; // declared but absent from this e-graph
            };
            if f.is_let_binding() {
                continue; // let-globals are references, not constructors
            }
            let schema = f.schema();
            if schema.output.name() != root.name() {
                return Err(PyValueError::new_err(format!(
                    "Constructor {name} outputs sort {} (expected {})",
                    schema.output.name(),
                    root.name()
                )));
            }
            let mut eq_mask = Vec::with_capacity(schema.input.len());
            for s in &schema.input {
                if s.is_eq_sort() {
                    if s.name() != root.name() {
                        return Err(PyValueError::new_err(UNSUPPORTED));
                    }
                    eq_mask.push(true);
                } else if s.is_container_sort() || s.name() != "String" {
                    return Err(PyValueError::new_err(UNSUPPORTED));
                } else {
                    eq_mask.push(false);
                }
            }
            let fid = funcs.len();
            funcs.push(FuncTab {
                term_name: name.clone(),
                eq_mask: eq_mask.clone(),
                arity: schema.input.len(),
            });

            let arity = schema.input.len();
            let mut rows: Vec<(Vec<EggValue>, EggValue)> = Vec::new();
            eg.function_for_each(name, |row| {
                if !row.subsumed && row.vals.len() == arity + 1 {
                    rows.push((row.vals[..arity].to_vec(), row.vals[arity]));
                }
            })
            .map_err(|e| PyValueError::new_err(format!("reading function {name}: {e}")))?;

            for (children, out) in rows {
                let mut ch_eq = Vec::new();
                let mut ch_prim = Vec::new();
                for (pos, (is_eq, v)) in eq_mask.iter().zip(children.iter()).enumerate() {
                    if *is_eq {
                        let cv = eg.get_canonical_value(*v, &root);
                        let i = class_index.get(&cv).copied().unwrap_or_else(|| {
                            let i = classes.len() as u32;
                            classes.push(cv);
                            class_index.insert(cv, i);
                            enodes_of.push(Vec::new());
                            i
                        });
                        ch_eq.push(i);
                    } else {
                        ch_prim.push((pos as u32, *v));
                    }
                }
                let out_c = eg.get_canonical_value(out, &root);
                let oi = class_index.get(&out_c).copied().unwrap_or_else(|| {
                    let i = classes.len() as u32;
                    classes.push(out_c);
                    class_index.insert(out_c, i);
                    enodes_of.push(Vec::new());
                    i
                });
                let e = Enode {
                    func: fid,
                    out: oi,
                    ch_eq,
                    ch_prim,
                    all_children: children,
                    head: *head,
                };
                enodes_of[oi as usize].push(enodes.len());
                enodes.push(e);
            }
        }

        let n = classes.len();
        // adjacency: parent class -> child classes (for Tarjan)
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for e in &enodes {
            for &c in &e.ch_eq {
                adj[e.out as usize].push(c);
            }
        }

        // ---- iterative Tarjan SCC; sccs emitted children-first ----
        let mut disc: Vec<u32> = vec![u32::MAX; n];
        let mut low: Vec<u32> = vec![0; n];
        let mut on_stack = vec![false; n];
        let mut tstack: Vec<u32> = Vec::new();
        let mut sccs: Vec<Vec<u32>> = Vec::new();
        let mut counter: u32 = 0;
        for start in 0..n {
            if disc[start] != u32::MAX {
                continue;
            }
            let mut call: Vec<(u32, usize)> = vec![(start as u32, 0)];
            while let Some(&mut (u, ref mut ei)) = call.last_mut() {
                if *ei == 0 {
                    disc[u as usize] = counter;
                    low[u as usize] = counter;
                    counter += 1;
                    tstack.push(u);
                    on_stack[u as usize] = true;
                }
                if *ei < adj[u as usize].len() {
                    let v = adj[u as usize][*ei];
                    *ei += 1;
                    if disc[v as usize] == u32::MAX {
                        call.push((v, 0));
                    } else if on_stack[v as usize] {
                        low[u as usize] = low[u as usize].min(disc[v as usize]);
                    }
                } else {
                    call.pop();
                    if let Some(&(p, _)) = call.last() {
                        low[p as usize] = low[p as usize].min(low[u as usize]);
                    }
                    if low[u as usize] == disc[u as usize] {
                        let mut comp = Vec::new();
                        loop {
                            let w = tstack.pop().unwrap();
                            on_stack[w as usize] = false;
                            comp.push(w);
                            if w == u {
                                break;
                            }
                        }
                        sccs.push(comp);
                    }
                }
            }
        }

        // ---- levelized DP over SCCs (children-first order) ----
        let mut costs: Vec<f64> = vec![f64::INFINITY; n];
        let mut best: Vec<Option<usize>> = vec![None; n];

        let enode_cost = |e: &Enode, costs: &Vec<f64>| -> Option<f64> {
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
            let noise = (fnv_hash(
                funcs[e.func].term_name.as_bytes(),
                &e.all_children,
            ) % 1024) as f64
                * noise_scale;
            if fold_max {
                Some(e.head + max_pos + eps * total + noise)
            } else {
                Some(e.head + total + noise)
            }
        };

        for comp in &sccs {
            let has_self_loop = comp.len() > 1
                || adj[comp[0] as usize]
                    .iter()
                    .any(|&c| c == comp[0]);
            if !has_self_loop {
                let v = comp[0] as usize;
                let mut bc = f64::INFINITY;
                let mut bi: Option<usize> = None;
                for &ei in &enodes_of[v] {
                    if let Some(c) = enode_cost(&enodes[ei], &costs) {
                        if c < bc {
                            bc = c;
                            bi = Some(ei);
                        }
                    }
                }
                costs[v] = bc;
                best[v] = bi;
            } else {
                let members: Vec<(u32, Vec<usize>)> = comp
                    .iter()
                    .map(|v| (*v, enodes_of[*v as usize].clone()))
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
                            if let Some(c) = enode_cost(&enodes[ei], &costs) {
                                if c < costs[vi] {
                                    costs[vi] = c;
                                    best[vi] = Some(ei);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(DagExtractor {
            funcs,
            enodes,
            classes,
            costs,
            best,
            memo: HashMap::default(),
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
        let Some(idx) = self.classes.iter().position(|c| *c == canonical) else {
            return Err(PyValueError::new_err("Unextractable root"));
        };
        let idx = idx as u32;
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
        self.classes.len()
    }

    /// Number of whitelisted enodes (debug/introspection).
    #[getter]
    fn num_enodes(&self) -> usize {
        self.enodes.len()
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
                    let arity = self.funcs[self.enodes[ei].func].arity;
                    if arity == 0 {
                        let name = self.funcs[self.enodes[ei].func].term_name.clone();
                        let t = termdag.0.app(name, Vec::new());
                        self.memo.insert(v, t);
                        results.push(t);
                        continue;
                    }
                    let mut slots: Vec<Option<TermId>> = vec![None; arity];
                    let mut to_enter: Vec<u32> = Vec::new();
                    {
                        let e = &self.enodes[ei];
                        let mask = &self.funcs[e.func].eq_mask;
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
                    let name = self.funcs[self.enodes[ei].func].term_name.clone();
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
