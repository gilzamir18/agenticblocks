"""Unit tests for _json_to_tool_calls — the JSON-as-text tool-call recovery path.

These exercise the pure function directly (dict + tool map -> _DummyToolCall list
or None), with no live model. The tool map mirrors what _parse_message builds:
    tool_name -> (all_params: set, required_params: set)
"""
import json

from agenticblocks.blocks.llm.agent import _json_to_tool_calls, _infer_tool_from_keys


# A representative tool set with distinct parameter shapes.
TOOLS = {
    "read_file": ({"file_path"}, {"file_path"}),
    "check_errors_js": ({"path"}, {"path"}),
    "edit_file": ({"path", "old_str", "new_str", "line"}, {"path", "old_str", "new_str"}),
}


def _names(calls):
    return [c.function.name for c in calls]


def _args(call):
    return json.loads(call.function.arguments)


# --- Format A/B: explicit tool name ---------------------------------------

def test_format_a_explicit_wrapper():
    data = {"tool_name": "read_file", "tool_args": {"file_path": "index.js"}}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["read_file"]
    assert _args(calls[0]) == {"file_path": "index.js"}


def test_format_b_name_plus_flat_params():
    data = {"name": "check_errors_js", "path": "index.js"}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["check_errors_js"]
    assert _args(calls[0]) == {"path": "index.js"}


def test_explicit_unknown_tool_name_returns_none():
    # A name that is not registered must never be invented/accepted.
    data = {"tool_name": "read_file_smart", "tool_args": {"path": "x"}}
    assert _json_to_tool_calls(data, TOOLS) is None


# --- Format C: bare params, inferred from real tool schemas ----------------

def test_format_c_infers_unique_tool_by_real_param_name():
    # "file_path" is unique to read_file.
    calls = _json_to_tool_calls({"file_path": "style.js"}, TOOLS)
    assert _names(calls) == ["read_file"]
    assert _args(calls[0]) == {"file_path": "style.js"}


def test_format_c_infers_edit_file_by_full_key_shape():
    data = {"path": "f.py", "old_str": "a", "new_str": "b"}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["edit_file"]


def test_format_c_ambiguous_returns_none():
    # Two single-param tools both declare exactly nothing that disambiguates an
    # empty/overlapping shape — ensure ambiguity yields None, never a guess.
    ambiguous_tools = {
        "tool_a": ({"path"}, {"path"}),
        "tool_b": ({"path"}, {"path"}),
    }
    assert _json_to_tool_calls({"path": "x"}, ambiguous_tools) is None


def test_format_c_missing_required_returns_none():
    # edit_file requires old_str/new_str; providing only a subset must not match
    # it, and "path" alone doesn't fit read_file (file_path) → None.
    assert _json_to_tool_calls({"path": "f.py"}, {"read_file": ({"file_path"}, {"file_path"})}) is None


def test_format_c_extra_unknown_keys_returns_none():
    # Keys not subset of any tool's params → no match.
    assert _json_to_tool_calls({"file_path": "x", "bogus": 1}, TOOLS) is None


# --- Format D: tool_calls list wrapper -------------------------------------

def test_format_d_nested_function_dict():
    data = {"tool_calls": [{"function": {"name": "read_file", "arguments": {"file_path": "x.js"}}}]}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["read_file"]
    assert _args(calls[0]) == {"file_path": "x.js"}


def test_format_d_function_as_plain_string():
    data = {"tool_calls": [{"function": "check_errors_js", "args": {"path": "x.js"}}]}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["check_errors_js"]


# --- Format E: fs_operations, type names a registered tool -----------------

def test_format_e_maps_type_to_registered_tool():
    data = {"fs_operations": [
        {"type": "edit_file", "path": "f.py", "old_str": "a", "new_str": "b", "line": 36},
        {"type": "read_file", "file_path": "g.py"},
    ]}
    calls = _json_to_tool_calls(data, TOOLS)
    assert _names(calls) == ["edit_file", "read_file"]
    # "type" is stripped; only real params survive.
    assert "type" not in _args(calls[0])
    assert _args(calls[0]) == {"path": "f.py", "old_str": "a", "new_str": "b", "line": 36}


def test_format_e_unknown_type_skipped():
    data = {"fs_operations": [{"type": "write_file", "path": "x", "content": "y"}]}
    # write_file is not registered → skipped → no results → None.
    assert _json_to_tool_calls(data, TOOLS) is None


# --- Non-tool JSON ---------------------------------------------------------

def test_plain_answer_json_returns_none():
    # The model emitting the actual answer payload must not be parsed as a tool call.
    assert _json_to_tool_calls({"records": []}, TOOLS) is None


def test_infer_helper_directly():
    assert _infer_tool_from_keys({"file_path"}, TOOLS) == "read_file"
    assert _infer_tool_from_keys({"path"}, TOOLS) == "check_errors_js"
    assert _infer_tool_from_keys({"nope"}, TOOLS) is None
