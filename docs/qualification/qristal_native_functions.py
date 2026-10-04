"""Exercise exact SDK native function source against retained CPython 3.10 Qristal.
This is native-function qualification, not Python 3.12 SDK package qualification.
"""
import ast
from dataclasses import dataclass
import hashlib,json,time
from collections.abc import Mapping
from pathlib import Path
source=Path('/sdk/marqov/simulation/executor.py').read_text()
@dataclass
class ExecutionResult:
 counts:dict
 backend:str
 execution_time_ms:float
 shots:int
 metadata:dict
scope=dict(ExecutionResult=ExecutionResult,hashlib=hashlib,json=json,time=time,Mapping=Mapping)
exec('from typing import Any\nfrom __future__ import annotations' if False else 'from typing import Any',scope)
exec(Path('/sdk/marqov/simulation/circuit_converter.py').read_text(),scope)
exec(Path('/sdk/marqov/simulation/config.py').read_text(),scope)
tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_import_qristal_core','_run_simulation')]
exec(compile(ast.Module(body=functions,type_ignores=[]),'exact-sdk-native-functions','exec'),scope)
results=[]
for gate,expected in [('x q[0];',{'10':32}),('x q[1];',{'01':32}),('h q[0];\ncx q[0],q[1];',None)]:
 qasm='OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'+gate+'\n'
 qasm=scope['ensure_measurements'](qasm)
 r=scope['_run_simulation'](qasm,32,scope['SimulationConfig'](backend_id='qpp',backend_type='statevector',num_qubits=2,seed=7))
 if expected is not None: assert r.counts==expected,r.counts
 else: assert set(r.counts)<={'00','11'} and sum(r.counts.values())==32,r.counts
 results.append({'counts':r.counts,'metadata':r.metadata})
print(json.dumps({'qualification':'exact-native-functions; retained Python 3.10 Qristal image; SDK package needs Python 3.12 qualification','passed':3,'results':results}))
