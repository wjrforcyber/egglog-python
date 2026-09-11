"""Tests for the e-graph cut iterator (`egglog.bindings.CutIterator`).

The CutIterator ports the structural half of ABC's `map` cut machinery to
the e-class graph: per-class cut sets union the merges of all e-nodes'
children cut sets, deduped by leaf set, dominance-filtered and capped, with
per-cut truth tables composed through per-operator kernels.  These tests pin:

* kernel truth-table parity with eggverse's `cost.operator_tt` convention,
* hand-computed cut structures on small Boolean e-graphs,
* end-to-end soundness: simulating `cut_term` over its fresh PIs must
  reproduce each cut's truth table bit for bit,
* cyclic-SCC degradation (self-loop classes become leaves, no hang),
* determinism of repeated enumeration.
"""

from __future__ import annotations

import pytest
from egglog import EGraph, Expr, StringLike, bindings, expr_parts, rewrite, ruleset, vars_

from egglog.egraph import to_runtime_expr


class Bool(Expr):
    @classmethod
    def var(cls, name: StringLike) -> Bool: ...

    @classmethod
    def zero(cls) -> Bool: ...

    @classmethod
    def one(cls) -> Bool: ...

    def __and__(self, other: Bool) -> Bool: ...

    def __or__(self, other: Bool) -> Bool: ...

    def __invert__(self) -> Bool: ...

    def aoi21(self, a: Bool, b: Bool) -> Bool: ...

    def mux(self, a: Bool, b: Bool) -> Bool: ...


_METHOD_TO_OP = {
    "var": "var",
    "zero": "zero",
    "one": "one",
    "__and__": "and",
    "__or__": "or",
    "__invert__": "not",
    "aoi21": "aoi21",
    "mux": "mux",
}


def _op_map(egraph: EGraph) -> dict[str, str]:
    out: dict[str, str] = {}
    for meth, op in _METHOD_TO_OP.items():
        fn = getattr(Bool, meth)
        _, egg_name = egraph._callable_to_egg(fn)
        out[egg_name] = op
    return out


def _sort_egg_name(egraph: EGraph, expr) -> str:
    tp = to_runtime_expr(expr).__egg_typed_expr__.tp
    return egraph._state.type_ref_to_egg(tp)


def _value(egraph: EGraph, expr):
    re_ = to_runtime_expr(expr)
    return egraph._state.typed_expr_to_value(re_.__egg_typed_expr__)


# ---------------- truth-table helpers (eggverse.truth convention) ------------


def _vm(i: int, n: int) -> int:
    return sum(1 << m for m in range(1 << n) if (m >> i) & 1)


def _not_(tt: int, n: int) -> int:
    return ~tt & ((1 << (1 << n)) - 1)


EXPECTED_TT = {
    "not": _not_(_vm(0, 1), 1),
    "and": _vm(0, 2) & _vm(1, 2),
    "or": _vm(0, 2) | _vm(1, 2),
    "xor": _vm(0, 2) ^ _vm(1, 2),
    "nand": _not_(_vm(0, 2) & _vm(1, 2), 2),
    "nor": _not_(_vm(0, 2) | _vm(1, 2), 2),
    "xnor": _not_(_vm(0, 2) ^ _vm(1, 2), 2),
    "aoi21": _not_((_vm(0, 3) & _vm(1, 3)) | _vm(2, 3), 3),
    "aoi22": _not_((_vm(0, 4) & _vm(1, 4)) | (_vm(2, 4) & _vm(3, 4)), 4),
    "oai21": _not_((_vm(0, 3) | _vm(1, 3)) & _vm(2, 3), 3),
    "oai22": _not_((_vm(0, 4) | _vm(1, 4)) & (_vm(2, 4) | _vm(3, 4)), 4),
    "mux": (_vm(0, 3) & _vm(1, 3)) | (_not_(_vm(0, 3), 3) & _vm(2, 3)),
}


def _eval_term(expr, k: int) -> int:
    """Exhaustively evaluate a Bool term over PIs n0..n{k-1} (bit m of the
    result = value at the minterm with PI i = bit i of m)."""

    def ev(decl):
        inner = getattr(decl, "expr", None)
        if inner is not None and (hasattr(inner, "callable") or hasattr(inner, "value")):
            decl = inner
        cb = getattr(decl, "callable", None)
        if cb is None:
            raise AssertionError(f"unexpected leaf {decl!r}")
        meth = cb.method_name
        args = list(getattr(decl, "args", ()))
        if meth == "var":
            a0 = args[0]
            while a0 is not None and not isinstance(a0, str):
                if hasattr(a0, "expr"):
                    a0 = a0.expr
                elif hasattr(a0, "value"):
                    a0 = a0.value
                else:
                    break
            name = str(a0)
            assert name.startswith("n"), repr(args[0])
            return _vm(int(name[1:]), k)
        kids = []
        for a in args:
            ae = getattr(a, "expr", None)
            kids.append(ev(ae if ae is not None else a))
        if meth == "zero":
            return 0
        if meth == "one":
            return (1 << (1 << k)) - 1
        if meth == "__and__":
            return kids[0] & kids[1]
        if meth == "__or__":
            return kids[0] | kids[1]
        if meth == "__invert__":
            return _not_(kids[0], k)
        if meth == "aoi21":
            return _not_((kids[0] & kids[1]) | kids[2], k)
        if meth == "mux":
            return (kids[0] & kids[1]) | (_not_(kids[0], k) & kids[2])
        raise AssertionError(f"unexpected method {meth}")

    return ev(expr_parts(expr).expr)


def _iter(egraph: EGraph, roots, k: int = 4, max_cuts: int = 16):
    sort = _sort_egg_name(egraph, roots[0] if roots else Bool.var("x"))
    return bindings.CutIterator(
        egraph._state.egraph,
        sort,
        _op_map(egraph),
        [_value(egraph, r) for r in roots],
        k,
        max_cuts,
    )


# --------------------------------- tests ------------------------------------


def test_op_truth_table_parity():
    """The Rust kernel table must equal the Python reference for every op."""
    for op, want in EXPECTED_TT.items():
        assert bindings.op_truth_table(op) == want, op
    with pytest.raises(ValueError, match="unknown operator"):
        bindings.op_truth_table("nope")


def test_cuts_simple_and():
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = x & y
    egraph.register(e)
    it = _iter(egraph, [e])
    cuts = it.cuts_of(egraph._state.egraph, _value(egraph, e), _sort_egg_name(egraph, e))
    assert len(cuts) == 2
    trivial, full = cuts
    assert trivial["enode"] is None and trivial["nnodes"] == 0
    assert len(full["leaves"]) == 2 and full["leaves"] == sorted(full["leaves"])
    assert full["tt"] == EXPECTED_TT["and"]
    assert full["enode"] is not None
    # the realizing term reproduces the cut truth table bit for bit
    td = bindings.TermDag()
    sort = _sort_egg_name(egraph, e)
    term = it.cut_term(egraph._state.egraph, td, _value(egraph, e), 1, sort)
    expr = egraph._from_termdag(td, term, to_runtime_expr(e).__egg_typed_expr__.tp)
    assert _eval_term(expr, 2) == EXPECTED_TT["and"]


def test_cut_term_soundness_exhaustive():
    """For every cut of every class: simulate the realizing term (leaves as
    fresh PIs n0..n{k-1}) and require bit-exact agreement with the cut TT."""
    egraph = EGraph()
    x, y, z = Bool.var("x"), Bool.var("y"), Bool.var("z")
    exprs = [
        x,
        y,
        z,
        x & y,
        ~(x & y),
        (x & y) | z,
        x.aoi21(~y, z | x),
        x.mux(~(y | z), x & y),
    ]
    for e in exprs:
        egraph.register(e)
    # light saturation to create choice e-nodes (demorgan-style)
    a, b = vars_("a b", Bool)
    rs = ruleset(
        rewrite(~(a & b)).to(~a | ~b),
        rewrite(~a | ~b).to(~(a & b)),
        rewrite(a | (a & b)).to(a),
    )
    for _ in range(4):
        rep = egraph.run(1, ruleset=rs)
        if not rep.updated:
            break
    it = _iter(egraph, [])  # whole-graph mode: registered exprs span cones
    sort = _sort_egg_name(egraph, exprs[0])
    for e in exprs:
        v = _value(egraph, e)
        cuts = it.cuts_of(egraph._state.egraph, v, sort)
        assert cuts, "every class gets at least its trivial cut"
        assert cuts[0]["enode"] is None, "trivial cut first"
        for ci, c in enumerate(cuts):
            k = len(c["leaves"])
            assert k <= 4
            td = bindings.TermDag()
            term = it.cut_term(egraph._state.egraph, td, v, ci, sort)
            expr = _from_termdag(egraph, td, term, e)
            got = _eval_term(expr, k)
            assert got == c["tt"], (
                f"cut {ci} of {e}: sim={got:#x} tt={c['tt']:#x} leaves={c['leaves']}"
            )


def _from_termdag(egraph, td, term, template):
    """Turn a TermId back into a Bool expr via the shared TermDag."""
    tp = to_runtime_expr(template).__egg_typed_expr__.tp
    return egraph._from_termdag(td, term, tp)


def test_trivial_cut_invariants():
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = (x & y) | ~x
    egraph.register(e)
    it = _iter(egraph, [e])
    sort = _sort_egg_name(egraph, e)
    cuts = it.cuts_of(egraph._state.egraph, _value(egraph, e), sort)
    assert cuts[0]["enode"] is None
    assert len(cuts[0]["leaves"]) == 1
    assert cuts[0]["tt"] == 0b10  # a free variable over itself
    assert all(c["leaves"] == sorted(c["leaves"]) for c in cuts)


def test_cyclic_scc_degrades_to_leaf():
    """The chain-cycle fixture must not hang; cyclic-only classes keep only
    their trivial cut."""
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e1 = x & y
    egraph.register(e1)
    a, b = vars_("a b", Bool)
    rs = ruleset(
        rewrite(a & b).to((a & b) | b),
        rewrite((a & b) | b).to(a & b),
    )
    for _ in range(6):
        rep = egraph.run(1, ruleset=rs)
        if not rep.updated:
            break
    it = _iter(egraph, [e1])
    sort = _sort_egg_name(egraph, e1)
    cuts = it.cuts_of(egraph._state.egraph, _value(egraph, e1), sort)
    assert cuts and cuts[0]["enode"] is None
    st = it.stats
    assert st["num_cyclic_classes"] > 0
    # soundness still holds for every surviving cut
    for ci, c in enumerate(cuts):
        td = bindings.TermDag()
        term = it.cut_term(egraph._state.egraph, td, _value(egraph, e1), ci, sort)
        expr = _from_termdag(egraph, td, term, e1)
        assert _eval_term(expr, len(c["leaves"])) == c["tt"]


def test_determinism():
    egraph = EGraph()
    x, y, z = Bool.var("x"), Bool.var("y"), Bool.var("z")
    e = x.aoi21(~(x | y), z & y)
    egraph.register(e)
    sort = _sort_egg_name(egraph, e)
    v = _value(egraph, e)
    it1 = _iter(egraph, [e])
    it2 = _iter(egraph, [e])
    assert it1.cuts_of(egraph._state.egraph, v, sort) == it2.cuts_of(egraph._state.egraph, v, sort)


def test_stats_fields():
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = x & y
    egraph.register(e)
    it = _iter(egraph, [e])
    st = it.stats
    assert st["k"] == 4 and st["max_cuts"] == 16
    assert st["num_classes_cone"] >= 3
    assert st["num_pi_classes"] == 2
    assert st["total_nontrivial_cuts"] >= 1
    assert st["elapsed_ms"] >= 0


def test_root_not_in_graph_raises():
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = x.mux(y, x)  # mux not whitelisted below
    egraph.register(e)
    op_map = _op_map(egraph)
    op_map = {k: v for k, v in op_map.items() if v != "mux"}
    sort = _sort_egg_name(egraph, e)
    with pytest.raises(ValueError, match="root value not present"):
        bindings.CutIterator(
            egraph._state.egraph, sort, op_map, [_value(egraph, e)], 4, 16
        )


def test_cutmapper_delay_first_bookkeeping():
    """CutMapper: matching + DP bookkeeping on a toy library."""
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = (x & y) | ~x
    egraph.register(e)
    it = _iter(egraph, [e])
    cells = [
        ("and2", 2, EXPECTED_TT["and"], 3.0, 2.0),
        ("or2", 2, EXPECTED_TT["or"], 2.0, 2.0),
        ("inv", 1, EXPECTED_TT["not"], 1.0, 1.0),
        ("mux2", 3, EXPECTED_TT["mux"], 4.0, 3.0),
    ]
    cm = bindings.CutMapper(it, cells, "delay")
    st = cm.stats
    assert st["num_unmatched"] == 0, cm.unmatched_classes
    assert st["max_root_arrival"] > 0
    assert st["num_pi"] == 2
    assert set(cm.pi_names.values()) == {"x", "y"}
    dec = cm.decisions
    assert dec, "root class must have a decision"
    # every decision's inputs are emittable class ids (PI, const, or decided)
    ok_ids = set(cm.pi_names) | set(cm.const_classes) | set(dec)
    for d in dec.values():
        assert all(i in ok_ids for i in d["inputs"]), d
        assert d["arrival"] >= 0 and d["flow"] > 0


def test_cutmapper_area_vs_delay():
    """Objective switches the ranking key deterministically."""
    egraph = EGraph()
    x, y = Bool.var("x"), Bool.var("y")
    e = ~(x & y)  # demorgan twin exists after saturation-free: use rules
    egraph.register(e)
    a, b = vars_("a b", Bool)
    rs = ruleset(rewrite(~(a & b)).to(~a | ~b), rewrite(~a | ~b).to(~(a & b)))
    for _ in range(3):
        rep = egraph.run(1, ruleset=rs)
        if not rep.updated:
            break
    it = _iter(egraph, [e])
    cells = [
        ("and2", 2, EXPECTED_TT["and"], 3.0, 2.0),
        ("nand2", 2, EXPECTED_TT["nand"], 2.0, 1.5),
        ("or2", 2, EXPECTED_TT["or"], 5.0, 1.0),
        ("inv", 1, EXPECTED_TT["not"], 1.0, 1.0),
    ]
    cm_delay = bindings.CutMapper(it, cells, "delay")
    cm_area = bindings.CutMapper(it, cells, "area")
    assert cm_delay.stats["num_unmatched"] == 0
    assert cm_area.stats["num_unmatched"] == 0
    # two runs with the same objective agree (determinism)
    cm_delay2 = bindings.CutMapper(it, cells, "delay")
    assert cm_delay.decisions == cm_delay2.decisions
