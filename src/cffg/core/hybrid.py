"""D12 declaration identity, reference transport, and structural admission.

No numerical arrays or compiler addresses belong here. D12.2 and D12.5 in
DERIVATIONS define the regional scopes and identity adapters checked below.
"""

from dataclasses import replace

from cffg.core.ir import (
    HYBRID_APPROXIMATION,
    HybridAdapterSpec,
    HybridComponentSpec,
    HybridRegionSpec,
    HybridSeparatorSpec,
    Template,
)

ORDINARY_FACTOR_TYPES = frozenset({
    "categorical", "matrix_cat", "transition_mixture", "gaussian", "gaussian_linear",
})


def map_component(spec, rename=lambda name: name, variable=None):
    """Copy all structural references, including both kinds of port names."""
    variable = rename if variable is None else variable
    return replace(
        spec,
        name=rename(spec.name),
        separators=tuple(
            replace(s, name=rename(s.name), regime=variable(s.regime), state=variable(s.state))
            for s in spec.separators
        ),
        regions=tuple(
            replace(r, name=rename(r.name), factor=rename(r.factor), ports=tuple(map(rename, r.ports)))
            for r in spec.regions
        ),
        adapters=tuple(
            replace(a, name=rename(a.name), separator=rename(a.separator), variable=variable(a.variable))
            for a in spec.adapters
        ),
        equalities=tuple(map(rename, spec.equalities)),
    )


def component_to_dict(spec):
    """Encode every semantic field; only unordered collections are sorted."""
    return {
        "approximation": spec.approximation,
        "support_semantics": spec.support_semantics,
        "separators": {
            s.name: {"regime": s.regime, "state": s.state, "active_support": sorted(s.active_support)}
            for s in sorted(spec.separators, key=lambda s: s.name)
        },
        "regions": {
            r.name: {"kind": r.kind, "factor": r.factor, "ports": list(r.ports)}
            for r in sorted(spec.regions, key=lambda r: r.name)
        },
        "adapters": {
            a.name: {"separator": a.separator, "variable": a.variable}
            for a in sorted(spec.adapters, key=lambda a: a.name)
        },
        "equalities": sorted(spec.equalities),
    }


def component_from_dict(name, document):
    """No legacy migration can assign D12 meaning to an incomplete record."""
    prefix = f"hybrid component {name!r}"

    def fields(record, expected, owner):
        if not isinstance(record, dict) or set(record) != set(expected):
            raise ValueError(f"{prefix}: malformed {owner}; required fields {sorted(expected)}")
        return record

    def records(key, expected, constructor):
        values = document[key]
        if not isinstance(values, dict):
            raise ValueError(f"{prefix}: {key} must be a named mapping")
        result = []
        for key_name, record in values.items():
            fields(record, expected, f"{key} {key_name!r}")
            for field in ("active_support", "ports"):
                if field in record and not isinstance(record[field], (list, tuple)):
                    raise ValueError(f"{prefix}: {key_name!r} {field} must be an array")
            result.append(constructor(name=key_name, **record))
        return tuple(result)

    fields(document, ("approximation", "support_semantics", "separators", "regions", "adapters", "equalities"), "declaration")
    if document["approximation"] != HYBRID_APPROXIMATION:
        raise ValueError(f"{prefix}: unknown approximation {document['approximation']!r}")
    if not isinstance(document["equalities"], (list, tuple)):
        raise ValueError(f"{prefix}: equalities must be an array of factor names")
    try:
        return HybridComponentSpec(
            name=name,
            approximation=document["approximation"],
            support_semantics=document["support_semantics"],
            separators=records("separators", ("regime", "state", "active_support"), HybridSeparatorSpec),
            regions=records("regions", ("kind", "factor", "ports"), HybridRegionSpec),
            adapters=records("adapters", ("separator", "variable"), HybridAdapterSpec),
            equalities=document["equalities"],
        )
    except (TypeError, KeyError) as error:
        raise ValueError(f"{prefix}: malformed declaration: {error}") from error


def require_no_hybrid(t: Template, error_type=ValueError):
    """Guard production before discovery, array loading, or publication."""
    if t.hybrid_components:
        raise error_type(
            f"hybrid components {sorted(t.hybrid_components)!r}: {HYBRID_APPROXIMATION} "
            "production compilation and bundle publication are not implemented"
        )


def hybrid_problems(t: Template) -> list[str]:
    problems = []
    factor_owners = {}
    variable_owners = {}
    for key, component in t.hybrid_components.items():
        prefix = f"hybrid component {key!r}"
        if not isinstance(component, HybridComponentSpec):
            problems.append(f"{prefix}: expected HybridComponentSpec")
            continue
        if component.name != key or not isinstance(key, str) or not key:
            problems.append(f"{prefix}: mapping key must match a nonempty declaration name")
        if component.approximation != HYBRID_APPROXIMATION:
            problems.append(f"{prefix}: unknown approximation {component.approximation!r}")
        if component.support_semantics != "fixed":
            problems.append(f"{prefix}: only fixed active-support semantics are supported")
        invalid = False
        names = set()
        for kind, records, cls in (
            ("separator", component.separators, HybridSeparatorSpec),
            ("region", component.regions, HybridRegionSpec),
            ("adapter", component.adapters, HybridAdapterSpec),
        ):
            for record in records:
                if not isinstance(record, cls):
                    problems.append(f"{prefix}: malformed {kind} declaration")
                    invalid = True
                    continue
                if not isinstance(record.name, str) or not record.name or record.name in names:
                    problems.append(f"{prefix}: empty or duplicate declaration name {record.name!r}")
                    invalid = True
                else:
                    names.add(record.name)
        if invalid:
            continue
        for factor in (*component.equalities, *(r.factor for r in component.regions)):
            if not isinstance(factor, str):
                problems.append(f"{prefix}: factor reference must be a name")
                invalid = True
                continue
            if factor in factor_owners:
                problems.append(f"{prefix}: factor {factor!r} overlaps objective owner {factor_owners[factor]!r}")
            factor_owners[factor] = key
        if not invalid:
            local, variables = _component_problems(t, component)
            problems.extend(f"{prefix}: {problem}" for problem in local)
            for variable in variables:
                if variable in variable_owners:
                    problems.append(f"{prefix}: variable {variable!r} overlaps component {variable_owners[variable]!r}")
                variable_owners[variable] = key
    return problems


def _component_problems(t, spec):
    problems = []
    if not spec.separators or not spec.regions:
        problems.append("nonempty separators and factor regions are required")
    # Only declared equality ownership identifies copies inside a component.
    parent = {v: v for v in t.variables}

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    owned = set(spec.equalities) | {r.factor for r in spec.regions}
    for name in spec.equalities:
        factor = t.factors.get(name)
        if factor is None or factor.type != "equality" or factor.param_edges:
            problems.append(f"equality {name!r}: requires an existing equality factor")
            continue
        variables = tuple(factor.edges.values())
        if variables and all(v in parent for v in variables):
            for variable in variables[1:]:
                parent[root(variable)] = root(variables[0])

    separators = {}
    support_by_regime = {}
    for s in spec.separators:
        if not isinstance(s.regime, str) or not isinstance(s.state, str):
            problems.append(f"separator {s.name!r}: regime and state references must be names")
            continue
        z, x = t.variables.get(s.regime), t.variables.get(s.state)
        if z is None or x is None or z.family != "discrete" or x.family != "gaussian":
            problems.append(f"separator {s.name!r}: requires existing categorical regime and Gaussian state edges")
            continue
        support = s.active_support
        if not support or any(type(i) is not int or i < 0 or i >= z.cardinality for i in support):
            problems.append(f"separator {s.name!r}: active support must contain valid integer labels")
            continue
        if len(set(support)) != len(support):
            problems.append(f"separator {s.name!r}: duplicate active support labels")
            continue
        zr, xr = root(s.regime), root(s.state)
        if zr in support_by_regime and support_by_regime[zr] != frozenset(support):
            problems.append(f"separator {s.name!r}: shared regime has inconsistent fixed active support")
        support_by_regime[zr] = frozenset(support)
        separators[s.name] = (zr, xr)
    if len(separators) != len(spec.separators):
        return problems, set()
    if len(set(separators.values())) != len(separators):
        problems.append("duplicate hybrid scopes would create overlapping separator entropy owners")

    roots = {v for pair in separators.values() for v in pair}
    variables = {v for v in parent if root(v) in roots}
    for name in spec.equalities:
        factor = t.factors.get(name)
        if factor is not None and not set(factor.edges.values()) <= variables:
            problems.append(f"equality {name!r}: lies outside declared hybrid scopes")

    incidence = {name: [] for name in separators}
    for region in spec.regions:
        factor = t.factors.get(region.factor)
        if factor is None:
            problems.append(f"region {region.name!r}: unknown factor {region.factor!r}")
            continue
        if not all(isinstance(port, str) and port in separators for port in region.ports):
            problems.append(f"region {region.name!r}: unknown separator port")
            continue
        for port in region.ports:
            incidence[port].append(region.kind)
        if factor.param_edges:
            problems.append(f"region {region.name!r}: parameter edges are unsupported")
        ports = [separators[p] for p in region.ports]
        edges = {role: root(v) for role, v in factor.edges.items() if v in parent}
        expected = None
        if region.kind == "recurrence" and len(ports) == 2 and factor.type == "recurrence_link":
            (zi, x), (zj, hx) = ports
            expected = {"in": zi, "out": zj, "drive": x}
            if x != hx or zi == zj:
                problems.append(f"region {region.name!r}: recurrence requires (i,x) -> (j,x) scopes")
        elif region.kind == "dynamics" and len(ports) == 2 and factor.type == "gaussian_mixture_linear":
            (zj, x), (zy, y) = ports
            expected = {"in": x, "out": y, "selector": zj}
            if zj != zy or x == y:
                problems.append(f"region {region.name!r}: only destination-selected affine (j,x) -> (j,y) dynamics are supported")
        elif region.kind == "unary" and len(ports) == 1:
            z, x = ports[0]
            expected = {
                "categorical": {"out": z},
                "gaussian": {"out": x},
                "gaussian_mixture": {"selector": z, "out": x},
            }.get(factor.type)
        if expected is None:
            problems.append(f"region {region.name!r}: unsupported {region.kind!r} scope/type {factor.type!r}")
        elif expected != edges:
            problems.append(f"region {region.name!r}: factor incidence differs from declared {region.kind} ports; source/pair-selected channels are unsupported")
        domains = {(t.variables[s.regime].cardinality, t.variables[s.state].dim) for s in spec.separators if s.name in region.ports}
        if len(domains) > 1:
            problems.append(f"region {region.name!r}: separator regime counts and state dimensions must agree")
    for name, uses in incidence.items():
        if not uses:
            problems.append(f"separator {name!r}: has no factor region")

    adapters_by_variable = {}
    for a in spec.adapters:
        if not isinstance(a.separator, str) or not isinstance(a.variable, str):
            problems.append(f"adapter {a.name!r}: separator and variable references must be names")
            continue
        pair = separators.get(a.separator)
        variable = t.variables.get(a.variable)
        if pair is None or variable is None or variable.family not in ("discrete", "gaussian"):
            problems.append(f"adapter {a.name!r}: requires an existing separator and ordinary categorical/Gaussian copy")
            continue
        wanted = pair[0] if variable.family == "discrete" else pair[1]
        if root(a.variable) != wanted:
            problems.append(f"adapter {a.name!r}: ordinary copy is not wired to its declared separator")
        if a.variable in adapters_by_variable:
            problems.append(f"adapter {a.name!r}: ordinary copy overlaps adapter {adapters_by_variable[a.variable].name!r}")
        adapters_by_variable[a.variable] = a
        exterior = [f for f in t.factors.values() if f.name not in owned and a.variable in f.all_edges().values()]
        if len(exterior) + len(t.ports_of(a.variable)) != 1:
            problems.append(f"adapter {a.name!r}: copy needs exactly one ordinary factor or exposed port")

    exterior_factors = set()
    for factor in t.factors.values():
        touching = set(factor.all_edges().values()) & variables
        if factor.name in owned:
            continue
        if touching:
            exterior_factors.add(factor.name)
        for variable in touching:
            if variable not in adapters_by_variable:
                problems.append(f"factor {factor.name!r}: undeclared ordinary boundary at {variable!r}; add an identity adapter or explicit factor region")
    # Equality wiring outside an adapter cannot hide an unsupported neighbor
    # or turn two singleton ports into a valid association-preserving boundary.
    endpoints = {v: [] for v in t.variables}
    for factor in t.factors.values():
        for variable in factor.all_edges().values():
            if variable in endpoints:
                endpoints[variable].append(factor)
    reached = {}
    exterior_variables = set()
    for variable, adapter in adapters_by_variable.items():
        pending, seen = [variable], set()
        while pending:
            edge = pending.pop()
            if edge in seen:
                continue
            seen.add(edge)
            exterior_variables.add(edge)
            for factor in endpoints[edge]:
                if factor.name in owned:
                    continue
                exterior_factors.add(factor.name)
                reached.setdefault((adapter.separator, factor.name), set()).add(t.variables[variable].family)
                if factor.type == "equality":
                    pending.extend(v for v in factor.edges.values() if v in endpoints)
                elif factor.type not in ORDINARY_FACTOR_TYPES or factor.param_edges:
                    problems.append(f"adapter {adapter.name!r}: unsupported ordinary neighbor {factor.name!r}/association cut; requires an explicit hybrid region")
    for (separator, factor), legs in reached.items():
        if legs == {"discrete", "gaussian"}:
            problems.append(f"separator {separator!r}: factor {factor!r} supplies association across separate ordinary regime/state adapters")
    for port in t.ports.values():
        if port.variable in variables and port.variable not in adapters_by_variable:
            problems.append(f"port {port.name!r}: undeclared ordinary boundary; add an identity adapter")
    for constraint in t.constraints:
        if constraint.target in owned | exterior_factors | variables | exterior_variables:
            problems.append(f"constraint on {constraint.target!r}: {constraint.kind.value} is unsupported in this hybrid component")
    return problems, variables
