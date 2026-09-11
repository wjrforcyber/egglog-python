//! Shared e-class graph snapshot.
//!
//! `ClassGraph::build` enumerates a whitelist of constructors from a live
//! egglog `EGraph`, canonicalizes every child/output value through the
//! union-find, interns classes in first-touch order, and condenses the
//! resulting digraph with an iterative Tarjan SCC (components emitted
//! children-first).  Both the DAG extractor (cost DP) and the cut iterator
//! (ABC-mapper-style cut enumeration) consume the same snapshot.
//!
//! Constructor enumeration walks the whitelist in sorted-name order so class
//! numbering is a function of the e-graph content alone (not of HashMap
//! iteration order, which varies per process).

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*};

use egglog::Value as EggValue;

pub(crate) const UNSUPPORTED: &str =
    "DagExtractor supports a single eq-sort root, whitelisted constructors, and String primitives only";

#[derive(Clone)]
pub(crate) struct FuncTab {
    /// Term head used for reconstruction (the egg function name).
    pub(crate) term_name: String,
    /// Per child position: true if the child sort is the eq sort (a class).
    pub(crate) eq_mask: Vec<bool>,
    pub(crate) arity: usize,
}

#[derive(Clone)]
pub(crate) struct Enode {
    pub(crate) func: usize,
    /// Class index of the output.
    pub(crate) out: u32,
    /// Class indices of eq children (in eq-position order).
    pub(crate) ch_eq: Vec<u32>,
    /// (position, value) of primitive (String) children.
    pub(crate) ch_prim: Vec<(u32, EggValue)>,
    pub(crate) head: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct SccProfile {
    pub(crate) num_classes: usize,
    pub(crate) num_sccs: usize,
    pub(crate) singleton_acyclic_nodes: usize,
    pub(crate) self_loop_nodes: usize,
    pub(crate) multi_node_nodes: usize,
    pub(crate) max_size: usize,
    pub(crate) mean_size: f64,
    /// Population variance of SCC sizes (divide by k).
    pub(crate) var_size: f64,
    pub(crate) median_size: f64,
    /// SCC size -> number of SCCs of that size.
    pub(crate) histogram: Vec<(usize, usize)>,
}

impl SccProfile {
    /// Purely observational statistics over the computed SCCs; reads `sccs`
    /// and `adj` and influences nothing downstream.
    pub(crate) fn compute(sccs: &[Vec<u32>], adj: &[Vec<u32>]) -> Self {
        let num_classes: usize = sccs.iter().map(|c| c.len()).sum();
        let num_sccs = sccs.len();
        let mut sizes: Vec<usize> = Vec::with_capacity(num_sccs);
        let mut singleton_acyclic_nodes = 0usize;
        let mut self_loop_nodes = 0usize;
        let mut multi_node_nodes = 0usize;
        let mut histogram: HashMap<usize, usize> = HashMap::default();
        for comp in sccs {
            let size = comp.len();
            sizes.push(size);
            *histogram.entry(size).or_insert(0) += 1;
            if size == 1 {
                let v = comp[0] as usize;
                if adj[v].iter().any(|&c| c == comp[0]) {
                    self_loop_nodes += 1;
                } else {
                    singleton_acyclic_nodes += 1;
                }
            } else {
                multi_node_nodes += size;
            }
        }
        sizes.sort_unstable();
        let max_size = sizes.last().copied().unwrap_or(0);
        let mean_size = if num_sccs == 0 {
            0.0
        } else {
            num_classes as f64 / num_sccs as f64
        };
        let var_size = if num_sccs == 0 {
            0.0
        } else {
            sizes
                .iter()
                .map(|&s| {
                    let d = s as f64 - mean_size;
                    d * d
                })
                .sum::<f64>()
                / num_sccs as f64
        };
        let median_size = if num_sccs == 0 {
            0.0
        } else if num_sccs % 2 == 1 {
            sizes[num_sccs / 2] as f64
        } else {
            (sizes[num_sccs / 2 - 1] + sizes[num_sccs / 2]) as f64 / 2.0
        };
        let mut histogram: Vec<(usize, usize)> = histogram.into_iter().collect();
        histogram.sort_unstable_by_key(|(s, _)| *s);
        SccProfile {
            num_classes,
            num_sccs,
            singleton_acyclic_nodes,
            self_loop_nodes,
            multi_node_nodes,
            max_size,
            mean_size,
            var_size,
            median_size,
            histogram,
        }
    }

    pub(crate) fn to_py_dict(&self, py: Python<'_>) -> PyObject {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("num_classes", self.num_classes).unwrap();
        d.set_item("num_sccs", self.num_sccs).unwrap();
        d.set_item("singleton_acyclic_nodes", self.singleton_acyclic_nodes).unwrap();
        d.set_item("self_loop_nodes", self.self_loop_nodes).unwrap();
        d.set_item("multi_node_nodes", self.multi_node_nodes).unwrap();
        let cyclic = self.self_loop_nodes + self.multi_node_nodes;
        d.set_item("cyclic_nodes", cyclic).unwrap();
        d.set_item("singleton_acyclic_pct", pct(self.singleton_acyclic_nodes, self.num_classes)).unwrap();
        d.set_item("self_loop_pct", pct(self.self_loop_nodes, self.num_classes)).unwrap();
        d.set_item("multi_node_pct", pct(self.multi_node_nodes, self.num_classes)).unwrap();
        d.set_item("cyclic_pct", pct(cyclic, self.num_classes)).unwrap();
        d.set_item("max_size", self.max_size).unwrap();
        d.set_item("mean_size", self.mean_size).unwrap();
        d.set_item("var_size", self.var_size).unwrap();
        d.set_item("median_size", self.median_size).unwrap();
        let hist = pyo3::types::PyDict::new(py);
        for (s, c) in &self.histogram {
            hist.set_item(s, c).unwrap();
        }
        d.set_item("size_histogram", hist).unwrap();
        d.into_any().unbind()
    }
}

pub(crate) fn pct(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * part as f64 / total as f64
    }
}

#[derive(Clone)]
pub(crate) struct ClassGraph {
    pub(crate) funcs: Vec<FuncTab>,
    pub(crate) enodes: Vec<Enode>,
    pub(crate) classes: Vec<EggValue>,
    pub(crate) class_index: HashMap<EggValue, u32>,
    /// Class index -> indices into `enodes`.
    pub(crate) enodes_of: Vec<Vec<usize>>,
    /// Parent class -> child classes (for Tarjan / reverse walks).
    pub(crate) adj: Vec<Vec<u32>>,
    /// Tarjan components, children-first (a component's descendants always
    /// appear at a strictly smaller index unless they are the component
    /// itself).
    pub(crate) sccs: Vec<Vec<u32>>,
    /// Class index -> component index in `sccs`.
    pub(crate) scc_of: Vec<u32>,
}

impl ClassGraph {
    /// Build the snapshot for one eq-sort over a constructor whitelist with
    /// per-constructor head costs (costs are carried on the enodes; the cut
    /// iterator passes zeros).
    pub(crate) fn build(
        eg: &egglog::EGraph,
        sort: &str,
        heads: &HashMap<String, f64>,
    ) -> PyResult<Self> {
        let root = eg
            .get_sort_by_name(sort)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown sort {sort}")))?
            .clone();
        if !root.is_eq_sort() {
            return Err(PyValueError::new_err(format!("Root sort {sort} is not an eq sort")));
        }

        let mut funcs: Vec<FuncTab> = Vec::new();
        let mut enodes: Vec<Enode> = Vec::new();
        let mut class_index: HashMap<EggValue, u32> = HashMap::default();
        let mut classes: Vec<EggValue> = Vec::new();
        let mut enodes_of: Vec<Vec<usize>> = Vec::new();

        // Sorted whitelist: deterministic class numbering across processes.
        let mut names: Vec<&String> = heads.keys().collect();
        names.sort();
        for name in names {
            let head = &heads[name];
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
        let sccs = tarjan_scc(n, &adj);
        let mut scc_of = vec![0u32; n];
        for (si, comp) in sccs.iter().enumerate() {
            for &v in comp {
                scc_of[v as usize] = si as u32;
            }
        }

        Ok(ClassGraph {
            funcs,
            enodes,
            classes,
            class_index,
            enodes_of,
            adj,
            sccs,
            scc_of,
        })
    }

    /// Forward closure over children from `root` classes (the PO cone: all
    /// classes the roots depend on).  Follows ALL enodes, including cyclic
    /// edges (conservative).
    pub(crate) fn cone_mask(&self, roots: &[u32]) -> Vec<bool> {
        let n = self.classes.len();
        let mut mask = vec![false; n];
        let mut stack: Vec<u32> = Vec::with_capacity(roots.len());
        for &r in roots {
            if !mask[r as usize] {
                mask[r as usize] = true;
                stack.push(r);
            }
        }
        while let Some(v) = stack.pop() {
            for &c in &self.adj[v as usize] {
                if !mask[c as usize] {
                    mask[c as usize] = true;
                    stack.push(c);
                }
            }
        }
        mask
    }
}

/// Iterative Tarjan SCC over the class digraph; components are emitted
/// children-first (a component completes only after all of its descendants).
pub(crate) fn tarjan_scc(n: usize, adj: &[Vec<u32>]) -> Vec<Vec<u32>> {
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
    sccs
}
