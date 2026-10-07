"""CPU control-flow dry run of cand6 (explore -> coder -> LoopAgent(verifier x2) -> finalize) via the real Evaluator.
Same CLI as dry_run_mock.py; SCEN=A (pass 1 FAIL, pass 2 submits) or SCEN=B (both FAIL, finalize submits)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dry_run_mock as m
from google.genai import types
from google.adk.models.llm_response import LlmResponse
VC = []
SCEN = os.environ.get('SCEN', 'A')  # A: pass1 FAIL, pass2 submits. B: both FAIL -> finalize submits

class Mock(m.ScriptedGemma):
    async def generate_content_async(self, req, stream=False):
        s = m._sys_text(req); rs = m._fn_responses(req); n = len(rs)
        tools = sorted((req.tools_dict or {}).keys())
        patch = self.gold_patch if self.gold_patch.endswith('\n') else self.gold_patch + '\n'
        texts = [p.text for c in (req.contents or []) for p in (c.parts or []) if p.text]
        if not s and not tools:
            role, parts = 'summarizer', [types.Part(text='SUMMARY.')]
        elif 'READ-ONLY first stage' in s:
            role = 'explore'
            parts = [[m._call('run_command', command="grep -n 'def complete' -r src | head")],
                     [types.Part(text='PLAN: rename complete->reset in src/httpx/_parsers.py')]][min(n, 1)]
        elif 'sole CODER' in s:
            role = 'coder'
            sc = [[m._call('run_command', command="cat > /tmp/fix.patch << 'EOF'\n" + patch + "EOF\ngit apply /tmp/fix.patch; echo exit=$?")],
                  [types.Part(text='Changed src/httpx/_parsers.py; inline check passed.')]]
            parts = sc[min(n, len(sc) - 1)]
        elif 'VERIFY stage' in s:
            role = 'verifier'
            # n counts function responses visible to the verifier (include_contents none)
            fail = types.Part(text='FAIL: tests/test_parsers.py::x AssertionError')
            VC.append(1); n = len(VC) - 1
            if SCEN == 'A':
                sc = [[m._call('run_command', command='cd /workspace && git diff --stat')], [fail],
                      [m._call('run_command', command='cd /workspace && timeout 50 python -m pytest -x -q -p no:cacheprovider tests/test_parsers.py 2>&1 | tail -n 3')],
                      [m._call('submit_patch')], [types.Part(text='VERIFIED')]]
            else:
                sc = [[m._call('run_command', command='cd /workspace && git diff --stat')], [fail],
                      [m._call('get_status')], [fail]]
            parts = sc[min(n, len(sc) - 1)]
        elif 'verify stage has finished' in s:
            role = 'finalize'
            parts = [[m._call('submit_patch')], [types.Part(text='SUBMITTED')]][min(n, 1)]
        else:
            role, parts = 'unknown', [types.Part(text='??')]
        m.LOG.append({'role': role, 'n_fr': n, 'tools': tools, 'state_verify_report_in_sys': ('note (empty on the first pass): FAIL' in s),
                      'n_contents': len(req.contents or []), 'last_response': None, 'emits': [p.text[:40] if p.text else 'call:' + p.function_call.name for p in parts]})
        yield LlmResponse(content=types.Content(role='model', parts=parts),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1000, candidates_token_count=50, total_token_count=1050))

m.ScriptedGemma = Mock
m.main()
