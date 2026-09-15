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
from cffg.core import Template, validate
from cffg.core.serialize import from_json, structure_hash, to_json
t = Template('empty')
validate(t)
restored = from_json(to_json(t))
assert structure_hash(restored) == structure_hash(t)
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
