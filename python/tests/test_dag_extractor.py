"""Tests for the DAG-specialized extractor (`egglog.bindings.DagExtractor`).

The DagExtractor computes per-class minimum costs with Tarjan SCC
condensation + topological levelized DP instead of Bellman-Ford relaxation,
using a pure-Rust head-cost table (constructor name -> head cost) and a
"sum" or "max" fold.  These tests pin its target features:

* parity with the default (Bellman-Ford) Extractor on acyclic and cyclic
  (commutativity style) e-graphs,
* head-cost sensitivity,
* cyclic e-graphs extract without panicking,
* memoized repeated extraction, unextractable roots, and constructor
  whitelisting.
"""

from __future__ import annotations

import pytest
from egglog import EGraph, Expr, StringLike, bindings, expr_parts, rewrite, ruleset, vars_

from egglog.egraph import _CostModel, to_runtime_expr


class Num(Expr):
    @classmethod
    def var(cls, name: StringLike) -> Num: ...

    def __add__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...


def _head_table(egraph: EGraph, add: float, mul: float) -> dict[str, float]:
    heads: dict[str, float] = {}
    for meth, cost in (("var", 0.0), ("__add__", add), ("__mul__", mul)):
        fn = getattr(Num, meth)
        _, egg_name = egraph._callable_to_egg(fn)
        heads[egg_name] = cost
    return heads


def _sort_egg_name(egraph: EGraph, expr) -> str:
    tp = to_runtime_expr(expr).__egg_typed_expr__.tp
    return egraph._state.type_ref_to_egg(tp)


def _saturated_egraph(cyclic: bool) -> tuple[EGraph, object, object]:
    egraph = EGraph()
    x, y = Num.var("x"), Num.var("y")
    e1 = x + y
    e2 = (x * y) + x
    egraph.register(e1)
    egraph.register(e2)
    a, b = vars_("a b", Num)
    rules = [rewrite(a + b).to(b + a), rewrite((a + b) + a).to(a * (a + b))]
    if cyclic:
        rules.append(rewrite(a * b).to(b * a))
    rs = ruleset(*rules)
    for _ in range(6):
        rep = egraph.run(1, ruleset=rs)
        if not rep.updated:
            break
    return egraph, e1, e2


def _bf_extract(egraph: EGraph, po, heads: dict[str, float], termdag):
    def model(_eg, expr, children):
        d = expr_parts(expr).expr
        cb = getattr(d, "callable", None)
        if cb is None:
            return 0.0
        meth = getattr(cb, "method_name", None)
        fn = getattr(Num, meth, None) if meth else None
        if fn is None:
            raise ValueError(meth)
        _, egg_name = egraph._callable_to_egg(fn)
        h = heads[egg_name]
        return 0.0 if h == 0.0 else h + sum(children)

    egg_cm = _CostModel(model, egraph).to_bindings_cost_model()
    re_ = to_runtime_expr(po)
    v = egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)
    egg_sort = egraph._state.type_ref_to_egg(re_.__egg_typed_expr__.tp)
    bf = bindings.Extractor([egg_sort], egraph._state.egraph, egg_cm)
    return bf.extract_best(egraph._state.egraph, termdag, v, egg_sort)


def _dag_extract(egraph: EGraph, po, heads: dict[str, float], termdag, fold="sum"):
    sort = _sort_egg_name(egraph, po)
    de = bindings.DagExtractor(egraph._state.egraph, sort, heads, fold, 0.0, 1e-7)
    re_ = to_runtime_expr(po)
    v = egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)
    return de.extract_best(egraph._state.egraph, termdag, v, sort)


@pytest.mark.parametrize("cyclic", [False, True])
def test_dag_matches_bellman_ford(cyclic):
    """Cost-class parity on acyclic and cyclic e-graphs."""
    egraph, e1, e2 = _saturated_egraph(cyclic)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    td = bindings.TermDag()
    for po in (e1, e2):
        cost_bf, term_bf = _bf_extract(egraph, po, heads, td)
        cost_dag, term_dag = _dag_extract(egraph, po, heads, td)
        assert round(float(cost_bf), 3) == round(float(cost_dag), 3)
        assert td.to_string(term_bf)
        assert td.to_string(term_dag)


def test_dag_head_costs_change_cost():
    """Raising a head cost must (weakly) increase the extracted minimum."""
    egraph, e1, _ = _saturated_egraph(cyclic=True)
    td = bindings.TermDag()
    cheap = _head_table(egraph, add=1.0, mul=5.0)
    expensive = _head_table(egraph, add=100.0, mul=5.0)
    c1, _ = _dag_extract(egraph, e1, cheap, td)
    c2, _ = _dag_extract(egraph, e1, expensive, td)
    assert c2 >= c1
    assert c2 > c1  # e1 contains an add head in every representation


def test_dag_introspection_counts():
    egraph, e1, _ = _saturated_egraph(cyclic=False)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    de = bindings.DagExtractor(egraph._state.egraph, _sort_egg_name(egraph, e1), heads, "sum", 0.0, 1e-7)
    assert de.num_classes >= 2  # at least x, y classes
    assert de.num_enodes >= 2


def test_dag_repeated_extraction_is_memoized():
    egraph, e1, _ = _saturated_egraph(cyclic=True)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    sort = _sort_egg_name(egraph, e1)
    de = bindings.DagExtractor(egraph._state.egraph, sort, heads, "sum", 0.0, 1e-7)
    td = bindings.TermDag()
    re_ = to_runtime_expr(e1)
    v = egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)
    c1, t1 = de.extract_best(egraph._state.egraph, td, v, sort)
    c2, t2 = de.extract_best(egraph._state.egraph, td, v, sort)
    assert (c1, t1) == (c2, t2)


def test_dag_unknown_sort_raises():
    egraph, e1, _ = _saturated_egraph(cyclic=False)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    with pytest.raises(ValueError, match="Unknown sort"):
        bindings.DagExtractor(egraph._state.egraph, "no_such_sort", heads, "sum", 0.0, 1e-7)


def test_dag_whitelist_ignores_unlisted_constructors():
    """Removing a constructor from the whitelist removes its rows; if the
    root then has no finite-cost derivation, extraction raises."""
    egraph, e1, _ = _saturated_egraph(cyclic=False)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    heads.pop(next(k for k in heads if "add" in k))
    sort = _sort_egg_name(egraph, e1)
    de = bindings.DagExtractor(egraph._state.egraph, sort, heads, "sum", 0.0, 1e-7)
    td = bindings.TermDag()
    re_ = to_runtime_expr(e1)
    v = egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)
    with pytest.raises(ValueError, match="Unextractable"):
        de.extract_best(egraph._state.egraph, td, v, sort)


def test_dag_max_fold():
    """max fold: head + max(positive children | 0) + eps*sum."""
    egraph, e1, _ = _saturated_egraph(cyclic=False)
    heads = _head_table(egraph, add=2.0, mul=3.0)
    sort = _sort_egg_name(egraph, e1)
    de = bindings.DagExtractor(egraph._state.egraph, sort, heads, "max", 1e-6, 0.0)
    td = bindings.TermDag()
    re_ = to_runtime_expr(e1)
    v = egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)
    cost, term = de.extract_best(egraph._state.egraph, td, v, sort)
    # e1 = x + y, vars are free: 2 + max(0,0) + eps*0
    assert round(cost, 3) == 2.0
    assert td.to_string(term)
