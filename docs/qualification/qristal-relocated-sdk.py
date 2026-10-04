import asyncio,json
from importlib.metadata import version
from marqov.circuits import Circuit
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor
async def main():
 executor=SimulationExecutor(SimulationConfig(backend_id='qpp',backend_type='statevector',seed=7,remote_backend_database_path='/opt/qristal/install-core/remote_backends.yaml'))
 results=[]
 for circuit,expected in [(Circuit().x(0).h(1).h(1),{'10':32}),(Circuit().h(0).h(0).x(1),{'01':32}),(Circuit().h(0).cx(0,1),None)]:
  r=await executor.execute(circuit,shots=32)
  assert sum(r.counts.values())==32 and r.metadata['engine']=='qpp'
  if expected is not None: assert r.counts==expected,r.counts
  else: assert set(r.counts)<={'00','11'},r.counts
  results.append({'counts':r.counts,'metadata':r.metadata})
 print(json.dumps({'qualification':'full current installed SDK with native QPP on compiler Python 3.12','sdk_version':version('marqov'),'results':results,'passed':3,'hosted':False}))
asyncio.run(main())
