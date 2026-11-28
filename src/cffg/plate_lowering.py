"""Resolve nested plate wiring directly into the compiler's slot layout."""

from dataclasses import dataclass, replace

from cffg._immutable import freeze
from cffg.core import (
    Broadcast, Chain, ConstraintSpec, FactorSpec, PortRole, PortSpec, Template, VariableSpec, validate,
)
from cffg.core.compose import ComposeError
from cffg.core.ir import EDGE_CONSTRAINTS, AgentSpec, GaussianFilterSpec, GaussianPlanSpec
from cffg.core.equality_tree import equality_wiring
from cffg.core.plate_grid import PlateGrid
from cffg.core.plates import _owner_template, _prefix, _prepared_plates, _scoped_blob_refs, _scoped_learnables
from cffg.topology import BoundConstraint, ExteriorPort, FactorDeclaration, TopologyLayout, layout_addresses, validate_slot_addresses


@dataclass(frozen=True)
class PlateInputs:
    """Numeric packers read scoped owners and the already allocated topology."""

    name: str
    topology: TopologyLayout
    blobs: dict
    learnables: dict
    agent: AgentSpec | GaussianPlanSpec | GaussianFilterSpec | None

    @property
    def variables(self):
        return self.topology.edges

    @property
    def factors(self):
        return self.topology.factors

    @property
    def ports(self):
        return self.topology.ports

    @property
    def constraints(self):
        return self.topology.constraints

    @property
    def plates(self):
        return {}


def _constraint_prototype(c):
    detail = freeze(c.detail)
    return c if detail is c.detail else replace(c, detail=detail)


def _declare(table, names, plate, kind):
    collisions = sorted(name for name in names if name in table)
    if collisions:
        raise ComposeError(f"plate {plate!r}: {kind} name clash {collisions}")
    table.update(names)


class _PlateWiring:
    def __init__(self, template, depth):
        # Variable entries share the one domain declaration in the fragment until
        # slot layout gives the edge its name, domain and literal slot addresses.
        self.domains = dict(template.variables)
        self.factors = {
            name: FactorDeclaration(f.name, f.type, f.edges, f.param_edges, f.param_blobs, f.function)
            for name, f in template.factors.items()
        }
        self.ports = dict(template.ports)
        self.constraints = [_constraint_prototype(c) for c in template.constraints]
        plates = list(_prepared_plates(template, _PlateWiring, depth))
        owners = _owner_template(template, plates)
        self.blobs = owners.blobs
        self.learnables = owners.learnables
        for name, spec, fragment, private in plates:
            self.add(name, spec, fragment, private, depth)

    @property
    def variables(self):
        return self.domains

    def callback_template(self, source):
        """Materialize legacy callback arguments only for a user-supplied IR reader."""
        return Template(
            source.name,
            variables={name: VariableSpec(name, v.cardinality, v.dim, v.ef) for name, v in self.domains.items()},
            factors={name: FactorSpec(name, f.type, dict(f.edges), dict(f.param_edges), dict(f.param_blobs),
                                      f.function) for name, f in self.factors.items()},
            constraints=[ConstraintSpec(c.kind, c.target, c.detail) for c in self.constraints],
            ports={name: PortSpec(name, p.variable, p.role) for name, p in self.ports.items()},
            blobs=self.blobs, learnables=self.learnables, agent=source.agent, schema_version=source.schema_version,
        )

    def add(self, plate, spec, fragment, private, depth):
        grid = PlateGrid(spec.dims)
        prefixes = [_prefix(plate, index) for index in grid]
        port_names = set()
        factor_names = set()
        for index, prefix in zip(grid, prefixes, strict=True):
            _declare(self.domains, {f"{prefix}/{name}": v for name, v in fragment.variables.items()},
                     plate, "variables")
            for local, current, previous, kind in (
                (fragment.ports, port_names, self.ports, "ports"),
                (fragment.factors, factor_names, self.factors, "factors"),
            ):
                new = {f"{prefix}/{value.name}" for value in local.values()}
                collisions = sorted((new & current) | (new & previous.keys()))
                if collisions:
                    raise ComposeError(f"plate {plate!r}: {kind} name clash {collisions}")
                current.update(new)
            if private:
                _declare(self.blobs, _scoped_blob_refs(fragment, prefix, index, private, depth), plate, "blobs")
                _declare(self.learnables, _scoped_learnables(fragment, prefix, private), plate, "learnables")

        aliases = {}
        junctions = []

        def consume(prefix, local):
            port = fragment.ports[local]
            name = f"{prefix}/{port.name}"
            if name not in port_names:
                raise ComposeError(f"no port {name!r}")
            port_names.remove(name)
            return f"{prefix}/{port.variable}"

        for rule in spec.bonds:
            if isinstance(rule, Chain):
                for here, there in grid.chain(rule.axis, rule.offset):
                    a = consume(_prefix(plate, here), rule.port_a)
                    b = consume(_prefix(plate, there), rule.port_b)
                    if self.domains[a].domain_key != self.domains[b].domain_key:
                        raise ComposeError(f"bond {a!r}~{b!r}: domain mismatch")
                    aliases[b] = a
                    del self.domains[b]
            elif isinstance(rule, Broadcast):
                hub = f"{plate}/{rule.hub}"
                if hub in self.domains or hub in self.ports or hub in port_names:
                    raise ComposeError(f"plate {plate!r}: generated Broadcast hub {hub!r} already exists")
                branches = [consume(prefix, rule.port) for prefix in prefixes]
                _, links, added = equality_wiring(hub, branches, f"{plate}/_bcast/{rule.hub}")
                domain = self.domains[branches[0]]
                for link in links:
                    if link in self.domains:
                        raise ComposeError(f"equality tree for {hub!r}: variable {link!r} already exists")
                    self.domains[link] = domain
                for name, *_ in added:
                    if name in factor_names or name in self.factors:
                        raise ComposeError(f"equality tree for {hub!r}: factor {name!r} already exists")
                    factor_names.add(name)
                junctions.extend(added)
                if len(branches) == 1:
                    aliases[branches[0]] = hub
                    del self.domains[branches[0]]
                self.domains[hub] = domain

        def endpoint(prefix, local):
            name = f"{prefix}/{local}"
            return aliases.get(name, name)

        constraints = [_constraint_prototype(c) for c in fragment.constraints]
        for prefix in prefixes:
            for f in fragment.factors.values():
                name = f"{prefix}/{f.name}"
                self.factors[name] = FactorDeclaration(
                    name, f.type, {r: endpoint(prefix, v) for r, v in f.edges.items()},
                    {r: endpoint(prefix, v) for r, v in f.param_edges.items()},
                    {r: f"{prefix}/{b}" if b in private else spec.shared_name(b)
                     for r, b in f.param_blobs.items()}, f.function,
                )
            for c in constraints:
                target = endpoint(prefix, c.target) if c.kind in EDGE_CONSTRAINTS else f"{prefix}/{c.target}"
                declaration = c.declaration if isinstance(c, BoundConstraint) else c
                self.constraints.append(BoundConstraint(declaration, target))
            for port in fragment.ports.values():
                name = f"{prefix}/{port.name}"
                if name in port_names:
                    self.ports[name] = ExteriorPort(name, endpoint(prefix, port.variable), port.role)
        for name, inlet, left, right in junctions:
            self.factors[name] = FactorDeclaration(
                name, "equality", {"a": inlet, "b": left, "c": right}, {}, {},
            )
        for rule in spec.bonds:
            if isinstance(rule, Broadcast):
                name = f"{plate}/{rule.hub}"
                self.ports[name] = ExteriorPort(name, name, PortRole.GLUE)


def lower_plates(template):
    """Validate the compact source, bind its copied endpoints, then allocate final slots."""
    validate(template)
    wiring = _PlateWiring(template, 0)
    topology = layout_addresses(
        wiring.domains, wiring.factors, tuple(wiring.constraints), wiring.ports, use_edge_domains=True,
    )
    validate_slot_addresses(topology)
    inputs = PlateInputs(template.name, topology, wiring.blobs, wiring.learnables, template.agent)
    validate(inputs)
    from cffg.kernels import needs_source_template

    types = {f.type for f in wiring.factors.values()}
    if needs_source_template(types):
        return wiring.callback_template(template), topology
    return inputs, topology
