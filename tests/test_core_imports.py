"""The public facade must not pull numerical runtimes into core model editing."""

import subprocess
import sys


def test_core_validation_and_json_without_runtime_imports():
    subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'equinox'}:
            raise AssertionError('core imported runtime: ' + fullname)

sys.meta_path.insert(0, NoRuntime())
from cffg.core import BlobRef, FactorSpec, FunctionSpec, PortRole, PortSpec, Template, VariableSpec, validate
from cffg.core.serialize import from_json, structure_hash, to_json
t = Template('empty')
validate(t)
restored = from_json(to_json(t))
assert structure_hash(restored) == structure_hash(t)
t = Template('nonlinear',
    variables={v: VariableSpec(v, dim=1) for v in ('x', 'y')},
    factors={'f': FactorSpec('f', 'gaussian_nonlinear', {'in':'x', 'out':'y'},
        param_blobs={'parameters':'theta', 'cov':'noise'},
        function=FunctionSpec('cffg.mean.cubic', '1.0.0'))},
    ports={v: PortSpec(v, v, PortRole.GLUE) for v in ('x', 'y')},
    blobs={b: BlobRef(b, b+'.npy') for b in ('theta', 'noise')})
validate(t)
assert structure_hash(from_json(to_json(t))) == structure_hash(t)
assert not {'jax', 'jaxlib', 'equinox'} & sys.modules.keys()
"""], check=True)


def test_lazy_public_exports_retain_identity():
    import cffg
    from cffg.compiler import compile_template
    from cffg.engine import Engine
    from cffg.kernel_bindings import GaussianBinding

    assert cffg.Engine is Engine
    assert cffg.compile_template is compile_template
    assert cffg.GaussianBinding is GaussianBinding
    assert set(cffg.__all__) <= set(dir(cffg))
