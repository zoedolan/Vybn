from types import SimpleNamespace

from spark.harness.providers import default_introspect, execute_tool_calls


class Provider:
    def build_tool_result(self, call_id, text):
        return {"id": call_id, "text": text}


class Bash:
    def __init__(self):
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return "ran:" + command

    def restart(self):
        return "restart-ok"


def call(name, cid, arguments=None):
    return SimpleNamespace(name=name, id=cid, arguments=arguments or {})


def test_execute_bash_tool_call_serial():
    response = SimpleNamespace(tool_calls=[call("bash", "1", {"command": "echo ok"})])
    bash = Bash()
    results, interrupted = execute_tool_calls(response, bash, Provider())
    assert interrupted is False
    assert bash.commands == ["echo ok"]
    assert results == [{"id": "1", "text": "ran:echo ok"}]


def test_execute_introspect_tool_call():
    response = SimpleNamespace(tool_calls=[call("introspect", "i")])
    results, interrupted = execute_tool_calls(
        response, Bash(), Provider(), introspect=lambda: "state"
    )
    assert interrupted is False
    assert results == [{"id": "i", "text": "state"}]


def test_default_introspect_handles_missing_events(tmp_path):
    out = default_introspect(str(tmp_path))
    assert "events unavailable" in out
