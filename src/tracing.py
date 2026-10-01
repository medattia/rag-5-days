"""Optional tracing with Arize Phoenix (Day 5).

If Phoenix is installed and its server is running (python -m phoenix.server.main serve),
every agent step, search and LLM call is recorded and shown at http://localhost:6006.
If Phoenix is not installed, everything still runs; tracing is simply switched off.
"""
from contextlib import contextmanager

_tracer = None


def start_tracing(project_name="rag-5-days"):
    """Connect to the local Phoenix server and auto-record OpenAI-compatible and Groq calls."""
    global _tracer
    if _tracer is not None:
        return True
    try:
        from phoenix.otel import register
        provider = register(project_name=project_name, auto_instrument=True, batch=True, verbose=False)
        _tracer = provider.get_tracer("rag-5-days")
        print("Phoenix tracing ON -> open http://localhost:6006")
        return True
    except Exception as err:
        print(f"Phoenix tracing OFF ({str(err)[:80]})")
        return False


@contextmanager
def span(name, kind, input_value=None):
    """One step in the timeline. kind: AGENT, TOOL, CHAIN (OpenInference span kinds)."""
    if _tracer is None:
        yield None
        return
    attributes = {"openinference.span.kind": kind}
    if input_value is not None:
        attributes["input.value"] = str(input_value)
    with _tracer.start_as_current_span(name, attributes=attributes) as s:
        yield s


def set_output(s, value):
    if s is not None:
        s.set_attribute("output.value", str(value))
