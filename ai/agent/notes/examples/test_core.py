import concurrent.futures
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from core import (
    ApprovalStore, Budget, FakeRemote, JsonLineStream, OperationLedger,
    ProtocolError, Registry, ScriptedModel, Tool, ToolGateway, fingerprint,
    normalize_reply, one_shot_tool, recover, require_coverage, rrf, run_loop,
    select_groups, validate_pairs,
)

def call(cid="c1", name="port", arguments=None):
    return {"id":cid, "name":name, "arguments":arguments if arguments is not None else {}}
def response(calls=None, text="8080", **extra):
    return {"complete":True, "text":text, "tool_calls":calls or [], **extra}
def gateway(authorize=None):
    registry=Registry()
    registry.install([Tool("port","v1",(),lambda args: 8080)])
    return ToolGateway(registry.snapshot(),authorize)

class LoopTests(unittest.TestCase):
    def test_one_shot_uses_only_one_model_call(self):
        m=ScriptedModel([response([call()])])
        self.assertEqual(one_shot_tool(m,gateway(),"port")["value"],8080)
        self.assertEqual(len(m.requests),1)

    def test_loop_feeds_exact_call_result_pair(self):
        m=ScriptedModel([response([call()]),response()])
        output=run_loop(m,gateway(),"port")
        self.assertEqual(len(m.requests),2)
        self.assertEqual(m.requests[1][-1]["tool_call_id"],"c1")
        self.assertEqual(output["status"],"answer_proposed")
        validate_pairs(output["messages"])

    def test_plain_answer_does_not_execute_tool(self):
        g=gateway()
        self.assertEqual(run_loop(ScriptedModel([response()]),g,"port")["answer"],"8080")
        self.assertEqual(g.effects,0)

    def test_truncation_never_dispatches(self):
        g=gateway()
        with self.assertRaises(ProtocolError):
            run_loop(ScriptedModel([response([call()],complete=False)]),g,"port")
        self.assertEqual(g.effects,0)

    def test_unknown_tool_is_observation(self):
        result=run_loop(ScriptedModel([response([call(name="missing")]),response()]),gateway(),"q")
        self.assertEqual(result["messages"][2]["content"]["code"],"UNKNOWN_TOOL")

    def test_invalid_arguments_do_not_execute(self):
        g=gateway()
        self.assertEqual(g.execute(call(arguments={"extra":1}))["code"],"INVALID_ARGUMENTS")
        self.assertEqual(g.effects,0)

    def test_denial_does_not_execute(self):
        g=gateway(lambda n,a:False)
        self.assertEqual(g.execute(call())["status"],"denied")
        self.assertEqual(g.effects,0)

    def test_call_ids_unique_in_batch(self):
        with self.assertRaises(ProtocolError):
            normalize_reply(response([call(),call()]))

    def test_call_ids_unique_across_rounds(self):
        with self.assertRaises(ProtocolError):
            run_loop(ScriptedModel([response([call()]),response([call()])]),gateway(),"q")

    def test_step_limit_not_success(self):
        result=run_loop(ScriptedModel([response([call()])]),gateway(),"q",max_steps=1)
        self.assertEqual(result["reason"],"STEP_LIMIT")

    def test_precancel_makes_no_model_call(self):
        m=ScriptedModel([response()])
        self.assertEqual(run_loop(m,gateway(),"q",cancelled=lambda:True)["status"],"cancelled")
        self.assertEqual(m.requests,[])

    def test_cancel_after_response_creates_not_started_result(self):
        m=ScriptedModel([response([call()])]); g=gateway()
        answer=run_loop(m,g,"q",cancelled=lambda: bool(m.requests))
        self.assertEqual(answer["status"],"cancelled")
        self.assertEqual(g.effects,0)
        validate_pairs(answer["messages"])

    def test_refusal_is_not_answer(self):
        out=run_loop(ScriptedModel([response(refusal="denied")]),gateway(),"q")
        self.assertEqual(out["status"],"refused")

    def test_refusal_with_calls_rejected(self):
        with self.assertRaises(ProtocolError):
            normalize_reply(response([call()],refusal="denied"))

    def test_orphan_result_rejected(self):
        with self.assertRaises(ProtocolError):
            validate_pairs([{"role":"tool","tool_call_id":"missing"}])

    def test_missing_result_rejected(self):
        with self.assertRaises(ProtocolError):
            validate_pairs([{"role":"assistant","tool_calls":[call()]}])

    def test_live_authorization_not_cached_with_tool(self):
        allowed=[True];g=gateway(lambda n,a:allowed[0])
        self.assertEqual(g.execute(call())["status"],"ok")
        allowed[0]=False
        self.assertEqual(g.execute(call())["status"],"denied")

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.groups=[
            {"id":"instruction","tokens":3,"required":True},
            {"id":"pair","tokens":8,"priority":2},
            {"id":"query","tokens":3,"required":True},
            {"id":"extra","tokens":2,"priority":1}]

    def test_group_selection_is_atomic(self):
        self.assertEqual([g["id"] for g in select_groups(self.groups,8)],
                         ["instruction","query","extra"])

    def test_required_overflow_explicit(self):
        with self.assertRaises(ValueError):select_groups(self.groups,5)

    def test_selection_retains_source_order(self):
        self.assertEqual([g["id"] for g in select_groups(self.groups,14)],
                         ["instruction","pair","query"])

    def test_input_groups_not_mutated(self):
        before=copy.deepcopy(self.groups)
        chosen=select_groups(self.groups,14);chosen[0]["tokens"]=99
        self.assertEqual(before,self.groups)

    def test_negative_tokens_rejected(self):
        with self.assertRaises(ValueError):select_groups([{"id":"x","tokens":-1}],10)

    def test_rrf_fuses_overlap(self):
        self.assertEqual(rrf([["a","b"],["b","c"]])[0][0],"b")

    def test_rrf_no_duplicate_vote_within_one_list(self):
        self.assertAlmostEqual(dict(rrf([["a","a"]]))["a"],1/61)

    def test_missing_worker_not_filtered_as_success(self):
        with self.assertRaises(ValueError):
            require_coverage({"a","b"},[{"task_id":"a","verified":True},None])

    def test_duplicate_worker_rejected(self):
        with self.assertRaises(ValueError):
            require_coverage({"a"},[{"task_id":"a","verified":True}]*2)

    def test_verified_worker_set_complete(self):
        self.assertEqual(set(require_coverage({"a"},[{"task_id":"a","verified":True}])),{"a"})

class RuntimeTests(unittest.TestCase):
    def test_snapshot_survives_replacement(self):
        r=Registry();r.install([Tool("t","v1",(),lambda a:1)]);old=r.snapshot()
        r.install([Tool("t","v2",(),lambda a:2)])
        self.assertEqual(old["t"].execute({}),1)
        self.assertEqual(r.snapshot()["t"].execute({}),2)

    def test_late_remove_cannot_delete_new_generation(self):
        r=Registry();old=r.install([]);r.install([Tool("t","v2",(),lambda a:2)])
        self.assertFalse(r.remove(old));self.assertIn("t",r.snapshot())

    def test_duplicate_registration_rejected(self):
        t=Tool("t","v1",(),lambda a:1)
        with self.assertRaises(ValueError):Registry().install([t,t])

    def test_snapshot_is_read_only(self):
        r=Registry();r.install([])
        with self.assertRaises(TypeError):r.snapshot()["x"]=None

    def test_parallel_budget_cannot_overreserve(self):
        b=Budget(10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            ids=list(pool.map(lambda _:b.reserve(1),range(100)))
        self.assertEqual(sum(key is not None for key in ids),10)

    def test_settlement_releases_unused_part(self):
        b=Budget(10);key=b.reserve(8);b.settle(key,2)
        self.assertIsNotNone(b.reserve(8))

    def test_double_settlement_rejected(self):
        b=Budget(10);key=b.reserve(8);b.settle(key,2)
        with self.assertRaises(ValueError):b.settle(key,2)

    def test_overspend_is_not_hidden(self):
        b=Budget(10);key=b.reserve(5);b.settle(key,12)
        self.assertEqual(b.spent,12);self.assertIsNone(b.reserve(1))

    def test_approval_binds_final_identity(self):
        s=ApprovalStore();identity={"actor":"u","tool":"write","args":{"port":80},"policy":1}
        key=s.issue(identity);changed=copy.deepcopy(identity);changed["args"]["port"]=443
        self.assertFalse(s.consume(key,changed));self.assertTrue(s.consume(key,identity))

    def test_policy_revocation_changes_approval(self):
        s=ApprovalStore();key=s.issue({"policy":1})
        self.assertFalse(s.consume(key,{"policy":2}))

    def test_approval_single_use(self):
        s=ApprovalStore();key=s.issue({"x":1})
        self.assertTrue(s.consume(key,{"x":1}));self.assertFalse(s.consume(key,{"x":1}))

    def test_fingerprint_is_order_independent(self):
        self.assertEqual(fingerprint({"a":1,"b":2}),fingerprint({"b":2,"a":1}))

class StreamingTests(unittest.TestCase):
    def wire(self):
        events=[{"type":"start","id":"a","kind":"text"},
                {"type":"delta","id":"a","value":"端口 8080"},
                {"type":"item_done","id":"a"},{"type":"response_done"}]
        return ("\n".join(json.dumps(e,ensure_ascii=False) for e in events)+"\n").encode()

    def test_utf8_survives_every_byte_boundary(self):
        for split in range(1,len(self.wire())):
            s=JsonLineStream();s.feed(self.wire()[:split]);s.feed(self.wire()[split:])
            self.assertEqual(s.finish()["a"]["value"],"端口 8080")

    def test_item_end_is_not_response_end(self):
        s=JsonLineStream()
        for line in self.wire().splitlines(keepends=True)[:-1]:s.feed(line)
        with self.assertRaises(ProtocolError):s.finish()

    def test_late_delta_rejected(self):
        s=JsonLineStream();s.feed(self.wire())
        with self.assertRaises(ProtocolError):
            s.feed(b'{"type":"delta","id":"a","value":"x"}\n')

    def test_response_with_unfinished_item_rejected(self):
        s=JsonLineStream();s.feed(b'{"type":"start","id":"a","kind":"text"}\n')
        with self.assertRaises(ProtocolError):s.feed(b'{"type":"response_done"}\n')

    def test_final_partial_json_rejected(self):
        s=JsonLineStream();s.feed(b'{"type":')
        with self.assertRaises(ProtocolError):s.finish()

class RecoveryTests(unittest.TestCase):
    def test_crash_after_remote_commit_then_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            proc=subprocess.run([sys.executable,"-S",str(Path(__file__).with_name("crash_worker.py")),directory],
                                capture_output=True,text=True,timeout=10)
            self.assertEqual(proc.returncode,17,proc.stderr)
            ledger=OperationLedger(path);remote=FakeRemote(path)
            self.assertEqual(ledger.get("op-1")[0],"unknown")
            self.assertEqual(remote.count(),1)
            result=recover(ledger,remote,"op-1")
            self.assertEqual(result["object_id"],"draft-op-1")
            self.assertEqual(ledger.get("op-1")[0],"confirmed")
            recover(ledger,remote,"op-1")
            self.assertEqual(remote.count(),1)

    def test_absent_remote_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=OperationLedger(Path(directory));remote=FakeRemote(Path(directory))
            ledger.prepare("op-1",{"x":1});ledger.dispatched("op-1")
            self.assertEqual(recover(ledger,remote,"op-1")["status"],"needs_reconciliation")
            self.assertEqual(remote.count(),0)

    def test_same_operation_different_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=OperationLedger(Path(directory));ledger.prepare("op",{"x":1})
            with self.assertRaises(ValueError):ledger.prepare("op",{"x":2})

    def test_remote_enforces_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            remote=FakeRemote(Path(directory))
            remote.apply("op",{"x":1});remote.apply("op",{"x":1})
            self.assertEqual(remote.count(),1)
            with self.assertRaises(ValueError):remote.apply("op",{"x":2})

    def test_new_intent_can_have_identical_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            remote=FakeRemote(Path(directory))
            remote.apply("op-1",{"x":1});remote.apply("op-2",{"x":1})
            self.assertEqual(remote.count(),2)

    def test_receipt_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=OperationLedger(Path(directory));ledger.prepare("op",{})
            ledger.commit("op",{"receipt":"one"})
            with self.assertRaises(ValueError):ledger.commit("op",{"receipt":"two"})

if __name__=="__main__":
    unittest.main(verbosity=2)
