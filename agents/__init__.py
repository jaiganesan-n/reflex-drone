"""Session 2 agents. Importing this package registers ``onnx-arc`` and ``onnx-mlp`` in the
app's agent registry (``app.agents_base.load_agent`` imports it on demand). ONNX Runtime is
imported only when such an agent is built."""
from app.agents_base import register_agent


def _onnx_arc(arg: str):
    from .onnx_agents import OnnxArcAgent
    return OnnxArcAgent(arg)


def _onnx_mlp(arg: str):
    from .onnx_agents import OnnxMLPAgent
    return OnnxMLPAgent(arg)


register_agent("onnx-arc", _onnx_arc)
register_agent("onnx-mlp", _onnx_mlp)
