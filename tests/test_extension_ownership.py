"""Cross-seam ownership checks; existing factor tests supply numerical oracles."""

import ast
import inspect
from functools import cache
from pathlib import Path

import pytest

import cffg

ROOT = Path(cffg.__file__).parent
OWNER_PACKAGES = (
    "specialized",
    "registered",
    "resolution",
    "execution",
    "readouts",
)
OWNER_FILES = (
    "compiled.py",
    "topology.py",
    "assembly.py",
    "assembly_checks.py",
    "array_requirements.py",
    "compiler_errors.py",
    "kernel_bindings.py",
    "kernel_runtime.py",
    "schedule.py",
    "schedule_contracts.py",
    "schedule_program.py",
    "lifecycle.py",
    "mu3_lifecycle.py",
    "ledger.py",
    "ledger_terms.py",
    "runtime_types.py",
    "runtime_checks.py",
    "iteration.py",
    "iteration_types.py",
    "exact_strategy.py",
    "outer_strategy.py",
    "outer_structure.py",
    "outer_kernels.py",
    "outer_outcome.py",
    "run_results.py",
)
ORCHESTRATION = (
    "compiler.py",
    "engine.py",
    "schedule_program.py",
    "execution/synchronous.py",
)


@cache
def sources():
    return {str(p.relative_to(ROOT)): p.read_text() for p in ROOT.rglob("*.py")}


@cache
def tree(path):
    return ast.parse(sources()[path])


def owner_paths():
    return sorted(
        set(OWNER_FILES)
        | {path for path in sources() if path.split("/")[0] in OWNER_PACKAGES}
    )


def module_name(path):
    return "cffg." + path.removesuffix(".py").replace("/", ".").removesuffix(
        ".__init__"
    )


def imported_modules(path):
    package = module_name(path).split(".")
    if not path.endswith("/__init__.py"):
        package.pop()
    names = set()
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                if node.module:
                    base.extend(node.module.split("."))
                prefix = ".".join(base)
            else:
                prefix = node.module or ""
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def called_names(path):
    aliases = {}
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                package = module_name(path).split(".")
                if not path.endswith("/__init__.py"):
                    package.pop()
                base = package[: len(package) - node.level + 1]
                if node.module:
                    base.extend(node.module.split("."))
                prefix = ".".join(base)
            else:
                prefix = node.module or ""
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{prefix}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{resolve(node.value)}.{node.attr}"
        return ""

    return {
        resolve(node.func)
        for node in ast.walk(tree(path))
        if isinstance(node, ast.Call)
    }


def selection_literals(path):
    values = set()
    for node in ast.walk(tree(path)):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            expressions = (node.test,)
        elif isinstance(node, ast.Match):
            expressions = tuple(case.pattern for case in node.cases)
        else:
            continue
        values.update(
            child.value
            for expression in expressions
            for child in ast.walk(expression)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
    return values


def factor_type_literals(path):
    values = set()
    for node in ast.walk(tree(path)):
        if isinstance(node, ast.Compare):
            subject, expression = node, node
        elif isinstance(node, ast.Match):
            subject, expression = node.subject, node
        else:
            continue
        if any(
            isinstance(part, ast.Attribute)
            and part.attr == "type"
            or isinstance(part, ast.Name)
            and part.id in {"ftype", "factor_type"}
            for part in ast.walk(subject)
        ):
            values.update(
                part.value
                for part in ast.walk(expression)
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
    return values


def test_external_registration_identities_never_enter_production_sources():
    from gaussian_kernel_fixture import CUSTOM as gaussian
    from test_discrete_kernels import CUSTOM as discrete
    from test_kernels import CUSTOM as ef

    for path, text in sources().items():
        for name in (discrete, gaussian, ef, "spin_tilt", "joint_precision"):
            assert name not in text, (path, name)


@pytest.mark.parametrize("path", ORCHESTRATION)
def test_orchestration_never_interprets_original_factor_types(path):
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "type"
        for node in ast.walk(tree(path))
    ), path


@pytest.mark.parametrize(
    "path",
    (
        "kernel_runtime.py",
        "execution/registered.py",
        "registered/packing.py",
        "lifecycle.py",
        "ledger.py",
        "ledger_terms.py",
        "readouts/registered.py",
        "readouts/discrete.py",
        "readouts/gaussian.py",
        "readouts/exponential_family.py",
    ),
)
def test_same_arena_builtins_have_no_second_type_selected_numerics(path):
    ordinary = {
        "categorical",
        "matrix_cat",
        "transition_mixture",
        "gaussian",
        "gaussian_linear",
        "gaussian_linear_obs",
        "gamma",
        "beta",
        "poisson_obs",
        "bernoulli_obs",
    }
    assert not factor_type_literals(path) & ordinary, path
    # The representation label "gaussian" is not the Gaussian prior type.
    assert not selection_literals(path) & (ordinary - {"gaussian"}), path


def test_lower_owners_do_not_import_public_orchestration():
    forbidden = {"cffg.compiler", "cffg.engine", "cffg.colony"}
    for path in owner_paths():
        assert not imported_modules(path) & forbidden, path


def test_extracted_owner_dependencies_are_acyclic():
    paths = {module_name(path): path for path in owner_paths()}
    graph = {
        name: imported_modules(path) & paths.keys() for name, path in paths.items()
    }
    visited = set()

    def visit(name, active):
        assert name not in active, " -> ".join((*active, name))
        if name not in visited:
            for dependency in sorted(graph[name]):
                visit(dependency, (*active, name))
            visited.add(name)

    for name in graph:
        visit(name, ())


@pytest.mark.parametrize(
    "callee,owner",
    (
        ("cffg.recurrence.structured_recurrence_update", "execution/cross_domain.py"),
        ("cffg.hybrid_gfe.linear_update", "execution/composites.py"),
        ("cffg.hybrid_gfe.prior_update", "execution/composites.py"),
        ("cffg.recurrence.structured_recurrence_energy", "ledger_terms.py"),
        ("cffg.hybrid_gfe.linear_energy", "ledger_terms.py"),
        ("cffg.hybrid_gfe.prior_energy", "ledger_terms.py"),
    ),
)
def test_specialized_rules_have_one_runtime_or_objective_adapter(callee, owner):
    callers = {path for path in owner_paths() if callee in called_names(path)}
    assert callers == {owner}


def test_constraint_rule_tables_have_separate_execution_and_objective_owners():
    for table, expected in (
        ("CONSTRAINT_MESSAGES", "execution/constraints.py"),
        ("CONSTRAINT_ENERGIES", "ledger_terms.py"),
    ):
        callers = {
            path
            for path in owner_paths()
            if any(
                isinstance(n, ast.Attribute) and n.attr == table
                for n in ast.walk(tree(path))
            )
        }
        assert callers == {expected}


@pytest.mark.parametrize(
    "callee",
    (
        "cffg.specialized.mixed.lower_recurrence",
        "cffg.specialized.composites.lower_hybrid_gfe",
        "cffg.specialized.constraints.lower_constraints",
    ),
)
def test_specialized_completion_calls_each_declared_lowerer_once(callee):
    callers = {path for path in sources() if callee in called_names(path)}
    assert callers == {"specialized/__init__.py"}
    name = callee.rsplit(".", 1)[-1]
    calls = [
        n
        for n in ast.walk(tree("specialized/__init__.py"))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]
    assert len(calls) == 1


def test_recurrence_classification_has_one_discovery_owner():
    classifiers = {
        path
        for path in owner_paths()
        if "recurrence_link" in factor_type_literals(path)
    }
    assert classifiers == {"specialized/discovery.py"}


def test_specialized_classification_does_not_return_to_shared_orchestration():
    names = {
        "recurrence_link",
        "gaussian_mixture",
        "gaussian_mixture_linear",
        "exactly_k",
        "at_most_k",
        "alldifferent",
        "table",
    }
    for path in (
        *ORCHESTRATION,
        "schedule.py",
        "resolution/resolve.py",
        "resolution/sources.py",
    ):
        assert not factor_type_literals(path) & names, path
        # Support transport's "table" tag is not a constraint classification.
        assert not selection_literals(path) & (names - {"table"}), path


def test_scheduled_and_flat_registered_updates_share_the_same_callables():
    from cffg.execution import registered, synchronous
    from cffg.execution.owners import SCHEDULED_EXECUTORS

    assert SCHEDULED_EXECUTORS.discrete is registered.execute_discrete
    assert SCHEDULED_EXECUTORS.gaussian is registered.execute_gaussian
    assert SCHEDULED_EXECUTORS.exponential_family is registered.execute_ef
    calls = called_names("execution/synchronous.py")
    for function in ("execute_discrete", "execute_ef"):
        assert f"cffg.execution.registered.{function}" in calls
    assert inspect.signature(synchronous._gauss_sweep).parameters["execute"].default is registered.execute_gaussian
    assert inspect.signature(synchronous.make_sweep).parameters["_gaussian_executor"].default is registered.execute_gaussian
    assert "cffg.execution.registered.execute_gaussian" in called_names("execution/startup.py")
    assert synchronous.make_sweep.__module__ == "cffg.execution.synchronous"
