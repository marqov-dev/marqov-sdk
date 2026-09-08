"""Offline, disposable-runtime probe of a built wheel using synthetic code only.

Run with the wheel path as the only argument in a Python 3.12 environment with
the runtime dependencies installed. No provider or hosted API is contacted.
"""

import base64
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    import cloudpickle

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package_root = root / "installed"
        with zipfile.ZipFile(sys.argv[1]) as wheel:
            wheel.extractall(package_root)
        sys.path.insert(0, str(package_root))
        from marqov import task, workflow
        import marqov

        assert Path(marqov.__file__).is_relative_to(package_root)

        @workflow
        def program(value):
            offset = 0.25

            @task
            def add(x):
                return x + offset

            @task
            def join(values):
                return sum(values["nested"])

            a = add(value)
            b = add(value + 1)
            total = join({"nested": (a, b)})
            return {"answer": total, "repeat": [total, (a, "units")], "constant": 0.5}

        capture = program(1.5).capture()
        nodes = list(capture.graph.nodes.values())
        assert nodes[2].dependencies == [nodes[0].id, nodes[1].id]
        # Callable loading/execution happens in fresh child interpreters, with
        # the exact wheel first on their path, not the checkout or installed SDK.
        child_env = {**os.environ, "PYTHONPATH": str(package_root)}
        values = {}
        for level in capture.graph.get_execution_order():
            for node_id in level:
                node = capture.graph.nodes[node_id]
                payload = root / "task.bin"
                payload.write_bytes(
                    cloudpickle.dumps(
                        (base64.b64decode(node.func_ref), node.args, node.kwargs, values)
                    )
                )
                result = root / "result.bin"
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import cloudpickle,sys\n"
                        "fn,args,kwargs,values=cloudpickle.load(open(sys.argv[1],'rb'))\n"
                        "def resolve(v):\n"
                        " if isinstance(v,dict):\n"
                        "  if v.get('__proxy__') is True: return values[v['node_id']]\n"
                        "  return {k:resolve(x) for k,x in v.items()}\n"
                        " if isinstance(v,(list,tuple)): return type(v)(resolve(x) for x in v)\n"
                        " return v\n"
                        "value=cloudpickle.loads(fn)(*resolve(args),**resolve(kwargs))\n"
                        "cloudpickle.dump(value,open(sys.argv[2],'wb'))\n",
                        str(payload),
                        str(result),
                    ],
                    env=child_env,
                    check=True,
                    timeout=30,
                )
                values[node_id] = cloudpickle.loads(result.read_bytes())
        assert capture.resolve_output(values) == {
            "answer": 4.5,
            "repeat": [4.5, (1.75, "units")],
            "constant": 0.5,
        }
        # The typed output descriptor also survives its own artifact roundtrip.
        template = cloudpickle.loads(cloudpickle.dumps(capture.output))
        assert template == capture.output
        print("Built-wheel capture, closures, recursive dependencies and return shape passed")


if __name__ == "__main__":
    main()
