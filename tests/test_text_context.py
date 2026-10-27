"""Text-context analyzer, judgment mapping, and the resize() hook.

Model weights are not required. Baseline equality uses the JSON produced by
the unmodified engine on 2026-10-21, except the Level 2 circle, which is
scaled uniformly onto the canvas.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.resize import _text_overlap_pairs, resize
from engine.text_context import facts, layout_bridge, schema
from engine.text_context.analyzer import analyze, judgments_for_resize
from engine.text_context.backends import BackendError, OllamaBackend

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "baseline"
INPUTS = {
    "chameleon_sample_level1_warmup_output.json": "chameleon_sample_level1_warmup.json",
    "chameleon_sample_level2_core_output.json": "chameleon_sample_level2_core.json",
    "chameleon_sample_level3_advanced_output.json": "chameleon_sample_level3_advanced.json",
}


def _text(eid, content, size, y, w=400, h=40):
    return {
        "id": eid, "type": "text", "x": 80, "y": y, "width": w, "height": h,
        "content": content, "font_family": "Solstice Sans", "font_size": size,
        "font_weight": "bold", "color": "rgba(255,255,255,1)", "text_align": "left",
    }


def _design(elements, w=1080, h=1080):
    return {"canvas": {"width": w, "height": h}, "elements": elements}


def _judgment(content, role, prominence, must_read, uncertainty="low"):
    return {
        "role": role, "prominence": prominence, "must_read": must_read,
        "related": [], "uncertainty": uncertainty, "why": "test", "content": content,
    }


class SchemaTests(unittest.TestCase):
    def test_rejects_unknown_duplicate_and_missing_ids(self):
        with self.assertRaises(schema.JudgmentError):
            schema.validate({"judgments": [{"id": "T9", "role": "headline", "prominence": "primary",
                                            "must_read": False, "related": [], "uncertainty": "low", "why": ""}]},
                            ["T1"], ["T1"])
        row = {"id": "T1", "role": "headline", "prominence": "primary", "must_read": False,
               "related": [], "uncertainty": "low", "why": "main line"}
        with self.assertRaises(schema.JudgmentError):
            schema.validate({"judgments": [row, dict(row)]}, ["T1"], ["T1"])
        with self.assertRaises(schema.JudgmentError):
            schema.validate({"judgments": []}, ["T1"], ["T1"])

    def test_a_call_to_action_marked_optional_is_corrected(self):
        from engine.text_context.schema import validate

        payload = {"judgments": [{
            "id": "T1", "role": "call_to_action", "prominence": "high", "must_read": False,
            "related": [], "uncertainty": "low", "why": "the button",
        }, {
            "id": "T2", "role": "badge", "prominence": "low", "must_read": True,
            "related": [], "uncertainty": "low", "why": "a sticker",
        }]}
        found, warnings = validate(payload, ["T1", "T2"], ["T1", "T2"])
        self.assertTrue(found["T1"]["must_read"])
        self.assertFalse(found["T2"]["must_read"])
        self.assertTrue(any("call_to_action" in note for note in warnings))
        self.assertTrue(any("decorative" in note for note in warnings))
        payload = {"judgments": [
            {"id": "T1", "role": "price", "prominence": "high", "must_read": True,
             "related": [{"id": "T2", "relation": "price_of"}], "uncertainty": "low", "why": "amount"},
            {"id": "T2", "role": "product_name", "prominence": "primary", "must_read": False,
             "related": [], "uncertainty": "medium", "why": "the thing"},
        ]}
        found, warnings = schema.validate(payload, ["T1", "T2"], ["T1", "T2"])
        self.assertEqual(set(found), {"T1", "T2"})
        self.assertEqual(found["T1"]["related"], [{"id": "T2", "relation": "price_of"}])
        self.assertEqual(warnings, [])


class FactTests(unittest.TestCase):
    def test_prompt_hides_source_ids_and_pixels(self):
        task = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())
        ctx = facts.build_context(task["source"], task["target_canvas"])
        prompt = ctx["prompt"]
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prompt))
        for element in task["source"]["elements"]:
            self.assertNotIn(element["id"], tokens)
        self.assertIn("SUNSET SALE", prompt)
        self.assertIn("TARGET canvas: 728x90", prompt)
        self.assertIn("Image pixels are NOT available", prompt)
        self.assertEqual(ctx["text_aliases"], ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"])
        self.assertIn("covers the whole canvas", prompt)

    def test_a_full_frame_contain_image_is_not_called_a_background(self):
        design = _design([
            {
                "id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "fit": "contain",
                "x": 0, "y": 0, "width": 1080, "height": 1080,
            },
            _text("title", "Weekend rack", 48, 80),
        ])
        prompt = facts.build_context(design, {"width": 1080, "height": 1080})["prompt"]
        self.assertNotIn("covers the whole canvas", prompt)


class AnalyzerTests(unittest.TestCase):
    def _task(self):
        return _design([_text("h", "Summer pears", 28, 200), _text("p", "$12", 72, 80)])

    def test_valid_reply_is_a_model_judgment(self):
        design = self._task()
        ctx = facts.build_context(design, {"width": 1080, "height": 1350})
        # aliases follow top-to-bottom: $12 then Summer pears
        reply = json.dumps({"judgments": [
            {"id": ctx["text_aliases"][0], "role": "price", "prominence": "high", "must_read": True,
             "related": [], "uncertainty": "low", "why": "the amount"},
            {"id": ctx["text_aliases"][1], "role": "product_name", "prominence": "primary", "must_read": False,
             "related": [], "uncertainty": "low", "why": "the product"},
        ]})

        class Fake:
            def identity(self):
                return {"model": "fake", "digest": "abc", "options": {"seed": 1}}

            def generate(self, system, user, response_schema):
                return reply, {"wall_s": 0.01, "done_reason": "stop"}

        sidecar = analyze(design, {"width": 1080, "height": 1350}, Fake(), cache_dir=None)
        self.assertEqual(sidecar["status"], "model")
        self.assertEqual(set(sidecar["judgments"]), {"h", "p"})
        self.assertEqual(sidecar["judgments"]["p"]["content"], "$12")
        self.assertEqual(sidecar["judgments"]["h"]["role"], "product_name")
        self.assertIsNotNone(judgments_for_resize(sidecar))

    def test_design_without_text_needs_no_inference(self):
        class NeverGenerate:
            def identity(self):
                raise AssertionError("no model identity should be requested")

            def generate(self, system, user, response_schema):
                raise AssertionError("no model judgment is needed")

        design = _design([{
            "id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "#ff6600",
            "x": 0, "y": 0, "width": 1080, "height": 1080,
        }])
        result = analyze(design, {"width": 728, "height": 90}, NeverGenerate(), cache_dir=None)
        self.assertEqual(result["status"], "no-text")
        self.assertEqual(result["text_ids"], [])
        self.assertIsNone(judgments_for_resize(result))

    def test_a_valid_cache_is_reused_and_an_empty_one_is_not(self):
        import tempfile

        design = self._task()
        target = {"width": 1080, "height": 1350}
        ctx = facts.build_context(design, target)
        reply = json.dumps({"judgments": [
            {"id": ctx["text_aliases"][0], "role": "price", "prominence": "high", "must_read": True,
             "related": [], "uncertainty": "low", "why": "the amount"},
            {"id": ctx["text_aliases"][1], "role": "product_name", "prominence": "primary", "must_read": False,
             "related": [], "uncertainty": "low", "why": "the product"},
        ]})

        class Fake:
            def __init__(self):
                self.calls = 0

            def identity(self):
                return {"model": "fake", "digest": "abc", "options": {"seed": 1}}

            def generate(self, system, user, response_schema):
                self.calls += 1
                return reply, {"wall_s": 0.01, "done_reason": "stop"}

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            backend = Fake()
            first = analyze(design, target, backend, cache_dir=cache)
            self.assertEqual(first["status"], "model")
            self.assertEqual(backend.calls, 1)
            second = analyze(design, target, backend, cache_dir=cache)
            self.assertEqual(second["status"], "cached_model")
            self.assertEqual(backend.calls, 1)
            self.assertEqual(set(second["judgments"]), {"h", "p"})
            key = next(cache.glob("*.json"))
            key.write_text("{}\n")
            third = analyze(design, target, backend, cache_dir=cache)
            self.assertEqual(third["status"], "model")
            self.assertEqual(backend.calls, 2)
            self.assertTrue(any("unusable" in item for item in third.get("warnings") or []))

    def test_bad_replies_fall_back_without_being_called_judgments(self):
        class Fake:
            def identity(self):
                return {"model": "fake", "digest": "abc", "options": {"seed": 1}}

            def generate(self, system, user, response_schema):
                return "not json", {"wall_s": 0.01, "done_reason": "stop"}

        sidecar = analyze(self._task(), {"width": 300, "height": 250}, Fake(), cache_dir=None, max_attempts=2)
        self.assertEqual(sidecar["status"], "fallback")
        self.assertEqual(sidecar["judgments"], {})
        self.assertEqual(len(sidecar["attempts"]), 2)
        self.assertIsNone(judgments_for_resize(sidecar))
        self.assertIn("no valid model reply", sidecar["fallback_reason"])

    def test_a_cached_optional_action_is_corrected_before_reuse(self):
        from engine.text_context.analyzer import _repair_cached_judgments

        cached = {
            "judgments": {
                "cta": {"role": "call_to_action", "must_read": False},
                "sticker": {"role": "badge", "must_read": True},
            },
            "warnings": [],
        }
        _repair_cached_judgments(cached)
        self.assertTrue(cached["judgments"]["cta"]["must_read"])
        self.assertFalse(cached["judgments"]["sticker"]["must_read"])
        self.assertEqual(len(cached["judgment_corrections"]), 2)

    def test_unreachable_backend_is_a_fallback(self):
        class Down:
            def identity(self):
                raise BackendError("ollama /api/show unreachable at http://127.0.0.1:1")

        sidecar = analyze(self._task(), {"width": 300, "height": 250}, Down(), cache_dir=None)
        self.assertEqual(sidecar["status"], "fallback")
        self.assertIn("unreachable", sidecar["fallback_reason"])

    def test_refuses_a_non_local_host(self):
        with self.assertRaises(BackendError):
            OllamaBackend(host="http://example.com:11434")

    def test_default_model_is_the_installed_provisional_model(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(OllamaBackend().model, "llama3.2:latest")


class ResizeHookTests(unittest.TestCase):
    def test_baseline_matches_saved_engine_output(self):
        for output_name, input_name in INPUTS.items():
            task = json.loads((ROOT / "input" / input_name).read_text())
            expected = json.loads((FIXTURES / output_name).read_text())
            got = resize(task["source"], task["target_canvas"])
            self.assertEqual(got, expected, input_name)
            self.assertEqual(layout_bridge.active_must_read_ids, set())
            self.assertIsNone(layout_bridge.last_guidance)

    def test_role_judgment_promotes_the_smaller_line_over_the_largest_font(self):
        design = _design([
            _text("sale", "50% OFF", 72, 80, h=80),
            _text("name", "Summer pears", 28, 220, h=50),
            _text("fine", "Void where prohibited.", 12, 900, w=800, h=24),
        ])
        target = {"width": 1080, "height": 1350}
        baseline = resize(design, target)
        judgments = {
            "sale": _judgment("50% OFF", "discount", "medium", False),
            "name": _judgment("Summer pears", "headline", "primary", False),
            "fine": _judgment("Void where prohibited.", "disclosure", "low", True),
        }
        guided = resize(design, target, judgments=judgments)
        base_size = {e["id"]: e["font_size"] for e in baseline["elements"]}
        new_size = {e["id"]: e["font_size"] for e in guided["elements"]}
        self.assertGreater(base_size["sale"], base_size["name"])
        self.assertGreater(new_size["name"], new_size["sale"])
        self.assertGreaterEqual(new_size["fine"], 8.0)
        self.assertEqual([e["content"] for e in guided["elements"]], [e["content"] for e in design["elements"]])
        self.assertEqual(layout_bridge.last_guidance.slots["name"], "headline")
        self.assertEqual(layout_bridge.last_guidance.slots["fine"], "disclosure")
        self.assertEqual(layout_bridge.active_must_read_ids, set())

    def test_high_uncertainty_keeps_the_heuristic(self):
        design = _design([_text("sale", "50% OFF", 72, 80, h=80), _text("name", "Summer pears", 28, 220)])
        target = {"width": 1080, "height": 1350}
        baseline = resize(design, target)
        judgments = {
            "sale": _judgment("50% OFF", "discount", "low", False, uncertainty="high"),
            "name": _judgment("Summer pears", "headline", "primary", False, uncertainty="high"),
        }
        guided = resize(design, target, judgments=judgments)
        self.assertEqual(
            {e["id"]: e["font_size"] for e in guided["elements"]},
            {e["id"]: e["font_size"] for e in baseline["elements"]},
        )

    def test_ungrouped_call_to_action_joins_the_banner_row(self):
        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        target = {"width": 728, "height": 90}
        baseline = resize(task["source"], target)
        judgments = {
            "headline": _judgment("Chase the Light", "headline", "primary", False),
            "cta": _judgment("Shop the Summer Line", "call_to_action", "high", True),
        }
        guided = resize(task["source"], target, judgments=judgments)
        base = {e["id"]: e for e in baseline["elements"]}
        new = {e["id"]: e for e in guided["elements"]}
        self.assertGreater(new["cta"]["font_size"], base["cta"]["font_size"])
        self.assertGreaterEqual(new["cta"]["font_size"], 8.0)
        self.assertEqual(new["cta"]["content"], "Shop the Summer Line")
        self.assertEqual(new["cta"]["font_family"], base["cta"]["font_family"])
        self.assertEqual(new["cta"].get("opacity", 1), 1)
        self.assertEqual([e["id"] for e in guided["elements"]], [e["id"] for e in task["source"]["elements"]])

    def test_short_strip_keeps_product_and_price_clear_of_rotated_ribbon(self):
        design = json.loads((ROOT / "eval" / "text_context" / "designs" / "d08.json").read_text())
        for index, element in enumerate(design["elements"]):
            element.setdefault("z_index", index)
        target = {"width": 728, "height": 90}
        judgments = {
            "t01": _judgment("50% OFF", "call_to_action", "high", False),
            "t02": _judgment("Weekend rack", "headline", "primary", True),
            "t03": _judgment("Ends Sunday", "date_time", "medium", True),
            "t04": _judgment("$39", "price", "high", True),
            "t05": _judgment("Cotton jersey, washed navy.", "body", "low", True),
        }
        baseline = resize(design, target)
        guided = resize(design, target, judgments=judgments)
        by_id = {element["id"]: element for element in guided["elements"]}
        self.assertEqual(layout_bridge.last_decision, "judged")
        self.assertGreaterEqual(by_id["t02"]["font_size"], 32)
        self.assertGreaterEqual(by_id["t04"]["font_size"], 32)
        self.assertFalse(_text_overlap_pairs(guided) - _text_overlap_pairs(baseline))
        self.assertEqual([element["id"] for element in guided["elements"]],
                         [element["id"] for element in design["elements"]])

    def test_must_read_copy_on_the_level3_banner_is_not_shrunk_away(self):
        task = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())
        source = task["source"]
        texts = {element["id"]: element["content"] for element in source["elements"] if element["type"] == "text"}
        judgments = {
            "ribbon_label": _judgment(texts["ribbon_label"], "call_to_action", "high", False),
            "headline": _judgment(texts["headline"], "headline", "primary", True),
            "subhead": _judgment(texts["subhead"], "body", "medium", True),
            "badge1_label": _judgment(texts["badge1_label"], "badge", "low", True),
            "badge2_label": _judgment(texts["badge2_label"], "badge", "low", True),
            "badge3_label": _judgment(texts["badge3_label"], "badge", "low", True),
            "cta_label": _judgment(texts["cta_label"], "call_to_action", "high", False),
            "legal_text": _judgment(texts["legal_text"], "disclosure", "low", True),
        }
        target = {"width": 728, "height": 90}
        baseline = resize(source, target)
        guided = resize(source, target, judgments=judgments)
        base = {element["id"]: element for element in baseline["elements"]}
        new = {element["id"]: element for element in guided["elements"]}
        self.assertEqual(layout_bridge.last_decision, "judged")
        self.assertEqual(layout_bridge.last_guidance.slots["ribbon_label"], "badge")
        self.assertEqual(layout_bridge.last_guidance.slots["cta_label"], "call_to_action")
        self.assertGreaterEqual(new["legal_text"]["font_size"], 8.0)
        self.assertGreaterEqual(new["subhead"]["font_size"], 8.0)
        self.assertGreater(new["headline"]["font_size"], new["legal_text"]["font_size"])
        self.assertGreater(new["cta_label"]["font_size"], new["ribbon_label"]["font_size"])
        self.assertGreaterEqual(new["legal_text"]["font_size"], base["legal_text"]["font_size"])
        rect_target = {"width": 300, "height": 250}
        rect_base = {element["id"]: element for element in resize(source, rect_target)["elements"]}
        rect = resize(source, rect_target, judgments=judgments)
        rect_text = {element["id"]: element for element in rect["elements"]}
        self.assertEqual(layout_bridge.last_decision, "judged")
        self.assertGreaterEqual(rect_text["legal_text"]["font_size"], 8.0)
        for badge_id in ("badge1_label", "badge2_label", "badge3_label"):
            self.assertGreaterEqual(rect_text[badge_id]["font_size"], rect_base[badge_id]["font_size"])
        self.assertGreaterEqual(rect_text["badge2_label"]["font_size"], 7.0)
        self.assertEqual(rect_text["legal_text"]["content"], texts["legal_text"])
        self.assertEqual(new["legal_text"]["content"], texts["legal_text"])
        self.assertEqual([element["id"] for element in guided["elements"]], [element["id"] for element in source["elements"]])
        self.assertFalse(_text_overlap_pairs(guided) - _text_overlap_pairs(baseline))


class ContractAndEntrypointTests(unittest.TestCase):
    def test_required_model_skips_inference_for_design_without_text(self):
        import io
        import tempfile
        import engine.main as main_mod

        class NeverGenerate:
            def identity(self):
                raise AssertionError("a textless design must not load the model")

            def generate(self, system, user, response_schema):
                raise AssertionError("a textless design must not call the model")

        source = _design([{
            "id": "bg", "type": "shape", "shape_type": "rectangle", "fill_color": "#ff6600",
            "x": 0, "y": 0, "width": 1080, "height": 1080,
        }])
        previous = os.environ.get("CHAMELEON_TEXT_MODE")
        os.environ["CHAMELEON_TEXT_MODE"] = "on"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                incoming, outgoing = Path(tmp) / "input", Path(tmp) / "output"
                incoming.mkdir()
                outgoing.mkdir()
                (incoming / "sample.json").write_text(json.dumps({
                    "source": source, "target_canvas": {"width": 728, "height": 90},
                }))
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "select_backend", return_value=(NeverGenerate(), "packaged-gguf")
                ), patch("sys.stdout", io.StringIO()):
                    code = main_mod.main()
                self.assertEqual(code, 0)
                self.assertTrue((outgoing / "sample_output.png").is_file())
                sidecar = json.loads((outgoing / "sample_sidecar.json").read_text())
                self.assertEqual(sidecar["status"], "no-text")
                self.assertFalse(sidecar["model_judgment"])
        finally:
            if previous is None:
                os.environ.pop("CHAMELEON_TEXT_MODE", None)
            else:
                os.environ["CHAMELEON_TEXT_MODE"] = previous

    def test_official_baselines_pass_the_hard_contract(self):
        from engine.contract import contract_failures, unresolved_warnings
        assets = ROOT / "brand_kit" / "assets"
        for output_name, input_name in INPUTS.items():
            task = json.loads((ROOT / "input" / input_name).read_text())
            output = json.loads((FIXTURES / output_name).read_text())
            self.assertEqual(contract_failures(task["source"], output, task["target_canvas"], assets), [], input_name)
            warnings = unresolved_warnings(task["source"], output, assets)
            if input_name.endswith("level3_advanced.json"):
                self.assertTrue(any("missing asset" in item for item in warnings))
                self.assertFalse(any("opacity" in item for item in warnings))

    def test_off_canvas_geometry_is_a_hard_failure(self):
        from engine.contract import contract_failures

        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        output = json.loads((FIXTURES / "chameleon_sample_level1_warmup_output.json").read_text())
        output["elements"][0]["x"] = -20
        failures = contract_failures(task["source"], output, task["target_canvas"], ROOT / "brand_kit" / "assets")
        self.assertTrue(any("off-canvas" in item for item in failures))
        output["elements"][0]["x"] = 0
        output["elements"][0]["rotation"] = 40
        output["elements"][0]["width"] = 400
        output["elements"][0]["height"] = 400
        failures = contract_failures(task["source"], output, {"width": 100, "height": 100}, ROOT / "brand_kit" / "assets")
        self.assertTrue(any("painted bounds" in item for item in failures))

    def test_changed_copy_is_a_hard_failure(self):
        from engine.contract import contract_failures
        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        output = json.loads((FIXTURES / "chameleon_sample_level1_warmup_output.json").read_text())
        output["elements"][2]["content"] = "rewritten"
        self.assertTrue(any("copy changed" in item for item in contract_failures(
            task["source"], output, task["target_canvas"], ROOT / "brand_kit" / "assets")))

    def test_missing_renderer_does_not_write_a_placeholder(self):
        import builtins
        import sys
        from engine.main import RendererUnavailable, render_to_png
        original = builtins.__import__
        removed = {}
        for key in list(sys.modules):
            if key == "renderer" or key.startswith("renderer."):
                removed[key] = sys.modules.pop(key)

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "renderer.converter.convert":
                raise ImportError("renderer hidden for test")
            return original(name, globals, locals, fromlist, level)

        try:
            with unittest.mock.patch("builtins.__import__", guarded):
                with self.assertRaises(RendererUnavailable):
                    render_to_png({"canvas": {"width": 10, "height": 10}, "elements": []}, ROOT / "brand_kit" / "assets", ROOT / "output" / "should-not-exist.png")
        finally:
            sys.modules.update(removed)

    def test_auto_without_a_packaged_model_is_the_no_model_path(self):
        from engine.text_context.entry import ModelRequiredError, select_backend
        previous = {key: os.environ.get(key) for key in (
            "CHAMELEON_TEXT_MODE", "CHAMELEON_TEXT_BACKEND", "CHAMELEON_USE_OLLAMA",
            "CHAMELEON_MODEL_PATH", "CHAMELEON_MODEL_DIR",
        )}
        os.environ["CHAMELEON_TEXT_MODE"] = "auto"
        os.environ["CHAMELEON_TEXT_BACKEND"] = "auto"
        os.environ["CHAMELEON_MODEL_PATH"] = "/tmp/chameleon-no-such-model.gguf"
        os.environ["CHAMELEON_MODEL_DIR"] = "/tmp/chameleon-no-such-models"
        os.environ.pop("CHAMELEON_USE_OLLAMA", None)
        try:
            backend, kind = select_backend()
            self.assertIsNone(backend)
            self.assertEqual(kind, "no-model")
            os.environ["CHAMELEON_TEXT_MODE"] = "on"
            with self.assertRaises(ModelRequiredError):
                select_backend()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_multiple_packaged_models_fail_instead_of_selecting_one_silently(self):
        import tempfile

        from engine.text_context.entry import ModelRequiredError, select_backend

        with tempfile.TemporaryDirectory() as folder:
            for name in ("a.gguf", "b.gguf"):
                (Path(folder) / name).write_bytes(b"GGUF")
            with patch.dict(os.environ, {
                "CHAMELEON_TEXT_MODE": "on",
                "CHAMELEON_TEXT_BACKEND": "auto",
                "CHAMELEON_MODEL_PATH": "",
                "CHAMELEON_MODEL_DIR": folder,
                "CHAMELEON_USE_OLLAMA": "",
            }):
                with self.assertRaisesRegex(ModelRequiredError, "Multiple GGUF files"):
                    select_backend()

    def test_model_selection_failure_clears_stale_task_artifacts(self):
        import io
        import tempfile

        from engine.main import main as engine_main
        import engine.main as main_mod
        from engine.text_context.entry import ModelRequiredError

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            incoming, outgoing = root / "input", root / "output"
            incoming.mkdir()
            outgoing.mkdir()
            (incoming / "sample.json").write_text("{}")
            for suffix in ("output.json", "output.png", "sidecar.json"):
                (outgoing / f"sample_{suffix}").write_bytes(b"stale")
            stderr = io.StringIO()
            with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(
                main_mod, "OUTPUT_DIR", outgoing
            ), patch.object(main_mod, "select_backend", side_effect=ModelRequiredError("ambiguous model")), patch(
                "sys.stderr", stderr
            ):
                code = engine_main()
            self.assertEqual(code, 2)
            self.assertEqual(list(outgoing.iterdir()), [])
            self.assertIn("ambiguous model", stderr.getvalue())

    def test_parallel_resizes_keep_their_own_judgments(self):
        import threading
        design = _design([
            _text("sale", "50% OFF", 72, 80, h=80),
            _text("name", "Summer pears", 28, 220, h=50),
        ])
        target = {"width": 1080, "height": 1350}
        left = {
            "sale": _judgment("50% OFF", "headline", "primary", False),
            "name": _judgment("Summer pears", "body", "low", False),
        }
        right = {
            "sale": _judgment("50% OFF", "discount", "medium", False),
            "name": _judgment("Summer pears", "headline", "primary", False),
        }
        sequential = {
            "left": {e["id"]: e["font_size"] for e in resize(design, target, judgments=left)["elements"] if e["type"] == "text"},
            "right": {e["id"]: e["font_size"] for e in resize(design, target, judgments=right)["elements"] if e["type"] == "text"},
        }
        barrier = threading.Barrier(2)
        found = {}

        def run(key, judgments):
            barrier.wait()
            output = resize(design, target, judgments=judgments)
            found[key] = {e["id"]: e["font_size"] for e in output["elements"] if e["type"] == "text"}

        threads = [threading.Thread(target=run, args=item) for item in (("left", left), ("right", right))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertGreater(sequential["left"]["sale"], sequential["left"]["name"])
        self.assertGreater(sequential["right"]["name"], sequential["right"]["sale"])
        self.assertEqual(found["left"], sequential["left"])
        self.assertEqual(found["right"], sequential["right"])

    def test_changed_type_and_color_are_hard_failures(self):
        from engine.contract import contract_failures, unresolved_warnings
        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        output = json.loads((FIXTURES / "chameleon_sample_level1_warmup_output.json").read_text())
        typed = json.loads(json.dumps(output))
        typed["elements"][2]["type"] = "shape"
        failures = contract_failures(task["source"], typed, task["target_canvas"], ROOT / "brand_kit" / "assets")
        self.assertTrue(any("type changed" in item for item in failures))
        colored = json.loads(json.dumps(output))
        colored["elements"][2]["color"] = "rgba(1,2,3,1)"
        failures = contract_failures(task["source"], colored, task["target_canvas"], ROOT / "brand_kit" / "assets")
        self.assertTrue(any("color changed" in item for item in failures))
        weighted = json.loads(json.dumps(output))
        weighted["elements"][2]["font_weight"] = "regular" if weighted["elements"][2].get("font_weight") != "regular" else "bold"
        self.assertTrue(any("font_weight changed" in item for item in contract_failures(
            task["source"], weighted, task["target_canvas"], ROOT / "brand_kit" / "assets")))
        rotated = json.loads(json.dumps(output))
        rotated["elements"][2]["rotation"] = 15
        hard = contract_failures(task["source"], rotated, task["target_canvas"], ROOT / "brand_kit" / "assets")
        self.assertFalse(any("rotation" in item for item in hard))
        warnings = unresolved_warnings(task["source"], rotated, ROOT / "brand_kit" / "assets")
        self.assertTrue(any(item.startswith("rotation changed") for item in warnings))

    def test_preserve_unresolved_fields_keeps_source_values(self):
        from engine.contract import preserve_unresolved_fields
        source = _design([_text("line", "Chase", 48, 40)])
        source["elements"][0]["opacity"] = 1
        source["elements"][0]["rotation"] = 0
        output = json.loads(json.dumps(source))
        output["elements"][0]["opacity"] = 0.4
        output["elements"][0]["z_index"] = 9
        notes = preserve_unresolved_fields(source, output)
        self.assertEqual(output["elements"][0]["opacity"], 1)
        self.assertNotIn("z_index", output["elements"][0])
        self.assertTrue(any("opacity" in note for note in notes))
        self.assertTrue(any("z_index" in note for note in notes))

    def _isolated_env(self):
        keys = (
            "CHAMELEON_TEXT_MODE", "CHAMELEON_TEXT_BACKEND", "CHAMELEON_USE_OLLAMA",
            "CHAMELEON_MODEL_PATH", "CHAMELEON_MODEL_DIR", "CHAMELEON_TEXT_NO_CACHE",
            "CHAMELEON_TEXT_CACHE_DIR",
        )
        return {key: os.environ.get(key) for key in keys}

    def _restore_env(self, previous):
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_required_model_inference_failure_publishes_nothing(self):
        import io
        import tempfile
        from engine.main import main as engine_main
        import engine.main as main_mod

        class Broken:
            def identity(self):
                return {"model": "broken", "digest": "abc", "options": {"seed": 1}}

            def generate(self, system, user, response_schema):
                raise BackendError("inference failed")

        previous = self._isolated_env()
        os.environ["CHAMELEON_TEXT_MODE"] = "on"
        os.environ.pop("CHAMELEON_USE_OLLAMA", None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming = root / "input"
            outgoing = root / "output"
            incoming.mkdir()
            outgoing.mkdir()
            task = {"source": _design([_text("line", "Chase the Light", 48, 80)]), "target_canvas": {"width": 300, "height": 250}}
            (incoming / "sample.json").write_text(json.dumps(task))
            (outgoing / "sample_output.json").write_text('{"stale": true}\n')
            (outgoing / "sample_output.png").write_bytes(b"stale-png")
            (outgoing / "sample_sidecar.json").write_text('{"status": "model"}\n')
            stderr = io.StringIO()
            try:
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "select_backend", return_value=(Broken(), "packaged-gguf")
                ), patch("sys.stderr", stderr):
                    code = engine_main()
            finally:
                self._restore_env(previous)
            self.assertEqual(code, 2)
            self.assertFalse((outgoing / "sample_output.json").exists())
            self.assertFalse((outgoing / "sample_output.png").exists())
            self.assertFalse((outgoing / "sample_sidecar.json").exists())
            self.assertIn("MODEL REQUIRED FAILED", stderr.getvalue())
            self.assertIn("inference failed", stderr.getvalue())

    def test_auto_fallback_is_labeled_and_contract_failure_publishes_nothing(self):
        import io
        import tempfile
        from engine.main import main as engine_main
        import engine.main as main_mod
        from engine.resize import resize as real_resize

        class Broken:
            def identity(self):
                return {"model": "broken", "digest": "abc", "options": {"seed": 1}}

            def generate(self, system, user, response_schema):
                raise BackendError("inference failed")

        previous = self._isolated_env()
        os.environ["CHAMELEON_TEXT_MODE"] = "auto"
        os.environ.pop("CHAMELEON_USE_OLLAMA", None)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming = root / "input"
            outgoing = root / "output"
            incoming.mkdir()
            outgoing.mkdir()
            task = {"source": _design([_text("line", "Chase the Light", 48, 80)]), "target_canvas": {"width": 728, "height": 90}}
            (incoming / "sample.json").write_text(json.dumps(task))

            def bad_resize(source, target, judgments=None):
                output = real_resize(source, target, judgments=judgments)
                output["elements"][0]["color"] = "rgba(1,2,3,1)"
                return output

            stderr = io.StringIO()
            try:
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "resize", bad_resize
                ), patch("sys.stderr", stderr):
                    code = engine_main()
            finally:
                self._restore_env(previous)
            self.assertEqual(code, 1)
            self.assertEqual(list(outgoing.iterdir()), [])
            self.assertIn("CONTRACT FAIL", stderr.getvalue())
            self.assertIn("color changed", stderr.getvalue())

        os.environ["CHAMELEON_TEXT_MODE"] = "auto"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incoming = root / "input"
            outgoing = root / "output"
            incoming.mkdir()
            outgoing.mkdir()
            task = {"source": _design([_text("line", "Chase the Light", 48, 80)]), "target_canvas": {"width": 400, "height": 400}}
            (incoming / "sample.json").write_text(json.dumps(task))
            stdout = io.StringIO()
            try:
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "select_backend", return_value=(Broken(), "packaged-gguf")
                ), patch("sys.stdout", stdout):
                    code = engine_main()
                sidecar = json.loads((outgoing / "sample_sidecar.json").read_text())
                published = json.loads((outgoing / "sample_output.json").read_text())
            finally:
                self._restore_env(previous)
            self.assertEqual(code, 0)
            self.assertEqual(sidecar["status"], "fallback")
            self.assertFalse(sidecar["model_judgment"])
            self.assertIn("fallback:", stdout.getvalue())
            self.assertTrue((outgoing / "sample_output.png").is_file())
            self.assertGreater((outgoing / "sample_output.png").stat().st_size, 32)
            self.assertEqual(published["elements"][0]["content"], "Chase the Light")

    def test_font_size_zero_fails_the_adobe_schema_and_publishes_nothing(self):
        import io
        import tempfile
        from engine.contract import adobe_schema_failures
        from engine.main import main as engine_main
        import engine.main as main_mod
        from engine.resize import resize as real_resize

        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        published = json.loads((FIXTURES / "chameleon_sample_level1_warmup_output.json").read_text())
        self.assertEqual(adobe_schema_failures(published), [])
        broken = json.loads(json.dumps(published))
        broken["elements"][2]["font_size"] = 0
        self.assertTrue(any("font_size" in item for item in adobe_schema_failures(broken)))

        previous = self._isolated_env()
        os.environ["CHAMELEON_TEXT_MODE"] = "off"
        with tempfile.TemporaryDirectory() as tmp:
            incoming = Path(tmp) / "input"
            outgoing = Path(tmp) / "output"
            incoming.mkdir()
            outgoing.mkdir()
            (incoming / "sample.json").write_text(json.dumps({
                "source": task["source"], "target_canvas": task["target_canvas"],
            }))

            def zero_font(source, target, judgments=None):
                output = real_resize(source, target, judgments=judgments)
                for element in output["elements"]:
                    if element.get("type") == "text":
                        element["font_size"] = 0
                        break
                return output

            stderr = io.StringIO()
            try:
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "resize", zero_font
                ), patch("sys.stderr", stderr):
                    code = engine_main()
            finally:
                self._restore_env(previous)
            self.assertEqual(code, 1)
            self.assertEqual(list(outgoing.iterdir()), [])
            self.assertIn("schema", stderr.getvalue())
            self.assertIn("font_size", stderr.getvalue())

    def test_invalid_cache_is_rejected_by_required_and_auto_modes(self):
        import io
        import tempfile
        from engine.main import main as engine_main
        import engine.main as main_mod
        from engine.text_context.analyzer import cache_key

        design = _design([_text("line", "Chase the Light", 48, 80)])
        target = {"width": 400, "height": 400}
        identity = {"model": "fake", "digest": "abc", "options": {"seed": 1}}

        class Counting:
            def __init__(self):
                self.calls = 0

            def identity(self):
                return identity

            def generate(self, system, user, response_schema):
                self.calls += 1
                raise BackendError("inference failed")

        def run(mode):
            backend = Counting()
            previous = self._isolated_env()
            os.environ["CHAMELEON_TEXT_MODE"] = mode
            os.environ.pop("CHAMELEON_USE_OLLAMA", None)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                incoming = root / "input"
                outgoing = root / "output"
                cache = root / "cache"
                incoming.mkdir()
                outgoing.mkdir()
                cache.mkdir()
                os.environ["CHAMELEON_TEXT_CACHE_DIR"] = str(cache)
                key = cache_key(design, target, identity, None)
                (cache / f"{key}.json").write_text("{}\n")
                (incoming / "sample.json").write_text(json.dumps({"source": design, "target_canvas": target}))
                stderr = io.StringIO()
                stdout = io.StringIO()
                try:
                    with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                        main_mod, "select_backend", return_value=(backend, "packaged-gguf")
                    ), patch("sys.stderr", stderr), patch("sys.stdout", stdout):
                        code = engine_main()
                    names = sorted(path.name for path in outgoing.iterdir())
                    text = stderr.getvalue() + stdout.getvalue()
                finally:
                    self._restore_env(previous)
            return code, names, text, backend.calls

        required_code, required_files, required_text, required_calls = run("on")
        self.assertEqual(required_code, 2)
        self.assertEqual(required_files, [])
        self.assertGreaterEqual(required_calls, 1)
        self.assertIn("MODEL REQUIRED FAILED", required_text)
        self.assertNotIn("cached_model", required_text)

        auto_code, auto_files, auto_text, auto_calls = run("auto")
        self.assertEqual(auto_code, 0)
        self.assertIn("sample_output.png", auto_files)
        self.assertGreaterEqual(auto_calls, 1)
        self.assertIn("fallback:", auto_text)
        self.assertNotIn("cached_model", auto_text)


class UnbreakableWordTests(unittest.TestCase):
    def test_a_long_token_fits_the_box_and_the_rendered_png(self):
        from PIL import Image
        import tempfile
        from engine.main import render_to_png
        from engine.resize import _get_font_manager, _font_style_for, _normalize_weight
        from renderer.converter.wrap import char_widths, wrap_text

        content = "https://example.com/stores/VeryLongSolsticeStoreName"
        source = _design([
            {
                "id": "bg", "type": "shape", "shape_type": "rectangle",
                "x": 0, "y": 0, "width": 1080, "height": 1080,
                "fill_color": "rgba(20,10,30,1)",
            },
            _text("store", content, 64, 40, w=900, h=80),
        ], w=1080, h=1080)
        target = {"width": 320, "height": 400}
        guided = resize(source, target, judgments={
            "store": _judgment(content, "headline", "primary", True),
        })
        element = next(item for item in guided["elements"] if item["id"] == "store")
        self.assertEqual(element["content"], content)
        manager = _get_font_manager()
        self.assertIsNotNone(manager)
        widths = char_widths(
            manager, element["font_family"], _normalize_weight(element.get("font_weight")),
            _font_style_for(element.get("font_weight"), element.get("font_style")),
            element["font_size"], content,
        )
        wrapped = wrap_text(content, widths, element["width"])
        self.assertNotIn("\n", wrapped)
        self.assertLessEqual(sum(widths), element["width"] + 0.5)
        baseline = resize(source, target)
        base = next(item for item in baseline["elements"] if item["id"] == "store")
        self.assertEqual(base["content"], content)
        base_widths = char_widths(
            manager, base["font_family"], _normalize_weight(base.get("font_weight")),
            _font_style_for(base.get("font_weight"), base.get("font_style")),
            base["font_size"], content,
        )
        self.assertLessEqual(sum(base_widths), base["width"] + 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "out.png"
            render_to_png(guided, ROOT / "brand_kit" / "assets", png)
            image = Image.open(png)
            self.assertEqual(image.size, (320, 400))
            self.assertGreater(png.stat().st_size, 100)
            colors = image.convert("RGB").getcolors(maxcolors=500000)
            self.assertIsNotNone(colors)
            self.assertGreater(len(colors), 1)

    def test_an_unbreakable_token_leaves_the_other_line_readable_on_a_strip(self):
        from engine.resize import _measured_lines

        token = "NorthwindDepotCodeAlphaOneTwo"
        phrase = "Reserve a table"
        source = _design([
            {
                "id": "bg", "type": "shape", "shape_type": "rectangle",
                "x": 0, "y": 0, "width": 1080, "height": 1080,
                "fill_color": "rgba(20,10,30,1)",
            },
            _text("token", token, 60, 200, w=800, h=120),
            _text("phrase", phrase, 30, 400, w=800, h=60),
        ])
        judgments = {
            "token": _judgment(token, "call_to_action", "high", True),
            "phrase": _judgment(phrase, "headline", "primary", True),
        }
        output = resize(source, {"width": 728, "height": 90}, judgments=judgments)
        by_id = {item["id"]: item for item in output["elements"]}
        for eid in ("token", "phrase"):
            element = by_id[eid]
            measured = _measured_lines(
                element["content"], element["font_family"], element.get("font_weight"),
                element.get("font_style"), element["font_size"], element["width"],
            )
            self.assertIsNotNone(measured)
            self.assertEqual(measured[0], 1)
        # 12 px is the gate's floor. The fitter rounds to 0.1 px, so a token
        # that fills its line can land a half pixel under that floor.
        self.assertGreaterEqual(by_id["token"]["font_size"], 11.5)
        self.assertGreaterEqual(by_id["phrase"]["font_size"], 16.0)
        self.assertGreater(by_id["phrase"]["font_size"], by_id["token"]["font_size"])


class CallToActionSlotTests(unittest.TestCase):
    def test_the_larger_action_keeps_the_button_when_a_smaller_pill_is_louder(self):
        from engine.text_context.layout_bridge import _pick_cta

        def pill(gid, label, width):
            return (gid, [
                {"id": gid + "-shape", "type": "shape", "x": 0, "y": 0, "width": width, "height": 40},
                {"id": label, "type": "text", "x": 0, "y": 0, "width": width, "height": 20, "content": label},
            ])

        large = pill("large", "shop", 200)
        small = pill("small", "ribbon", 80)
        usable = {
            "shop": {"role": "call_to_action", "prominence": "high", "must_read": False, "uncertainty": "low"},
            "ribbon": {"role": "call_to_action", "prominence": "primary", "must_read": True, "uncertainty": "low"},
        }
        chosen, badges, loose = _pick_cta(usable, [large, small], large, [small], None)
        self.assertIsNone(loose)
        self.assertEqual(chosen[0], "large")
        self.assertEqual([group[0] for group in badges], ["small"])

    def test_resize_keeps_both_action_roles(self):
        design = _design([
            {"id": "btn_bg", "type": "shape", "shape_type": "rectangle", "group_id": "btn",
             "x": 40, "y": 400, "width": 240, "height": 64, "rotation": 0, "fill_color": "#eeeeee"},
            _text("btn_label", "Come in", 22, 412, w=200, h=32),
            {"id": "flag_bg", "type": "shape", "shape_type": "rectangle", "group_id": "flag",
             "x": 700, "y": 40, "width": 160, "height": 40, "rotation": -12, "fill_color": "#f2c14e"},
            _text("flag_label", "Tonight", 16, 48, w=140, h=24),
        ])
        design["elements"][1]["group_id"] = "btn"
        design["elements"][3]["group_id"] = "flag"
        design["elements"][3]["rotation"] = -12
        judgments = {
            "btn_label": _judgment("Come in", "call_to_action", "high", True),
            "flag_label": _judgment("Tonight", "call_to_action", "high", True),
        }
        resize(design, {"width": 1080, "height": 1080}, judgments=judgments)
        self.assertEqual(judgments["btn_label"]["role"], "call_to_action")
        self.assertEqual(judgments["flag_label"]["role"], "call_to_action")
        self.assertEqual(layout_bridge.last_guidance.slots["btn_label"], "call_to_action")
        self.assertEqual(layout_bridge.last_guidance.slots["flag_label"], "badge")

    def test_level2_banner_shop_label_stays_one_line(self):
        from engine.resize import _measured_lines

        task = json.loads((ROOT / "input" / "chameleon_sample_level2_core.json").read_text())
        source = task["source"]
        texts = {item["id"]: item["content"] for item in source["elements"] if item["type"] == "text"}
        judgments = {
            "headline": _judgment(texts["headline"], "headline", "primary", True),
            "subhead": _judgment(texts["subhead"], "body", "medium", True),
            "badge_label": _judgment(texts["badge_label"], "call_to_action", "high", True),
            "cta_label": _judgment(texts["cta_label"], "call_to_action", "high", True),
        }
        guided = resize(source, {"width": 728, "height": 90}, judgments=judgments)
        label = next(item for item in guided["elements"] if item["id"] == "cta_label")
        headline = next(item for item in guided["elements"] if item["id"] == "headline")
        measured = _measured_lines(
            label["content"], label["font_family"], label.get("font_weight"), label.get("font_style"),
            label["font_size"], label["width"],
        )
        self.assertEqual(label["content"], texts["cta_label"])
        self.assertEqual(measured[0], 1)
        self.assertLessEqual(measured[1], label["width"] + 0.5)
        self.assertGreater(headline["font_size"], label["font_size"])


class RetiredRunTests(unittest.TestCase):
    def test_retired_command_writes_nothing(self):
        import io
        from engine.text_context.run import main as retired

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            code = retired(["--out", "output/should-not-exist"])
        self.assertEqual(code, 2)
        self.assertIn("retired", stdout.getvalue())
        self.assertFalse((ROOT / "output" / "should-not-exist").exists())


class TiltedGroupTests(unittest.TestCase):
    def _inside(self, element, canvas):
        from engine.resize import _rotated_bbox

        rx, ry, rw, rh = _rotated_bbox(
            element["x"], element["y"], element["width"], element["height"], element.get("rotation") or 0,
        )
        width, height = canvas["width"], canvas["height"]
        return rx >= -0.5 and ry >= -0.5 and rx + rw <= width + 0.5 and ry + rh <= height + 0.5

    def test_a_tilted_group_that_starts_inside_stays_inside_on_short_canvases(self):
        source = _design([
            {
                "id": "bg", "type": "shape", "shape_type": "rectangle",
                "x": 0, "y": 0, "width": 900, "height": 900,
                "fill_color": "rgba(20,10,30,1)",
            },
            _text("headline", "Evening market", 48, 360, w=520, h=80),
            {
                "id": "pill", "type": "shape", "shape_type": "rectangle",
                "x": 560, "y": 80, "width": 240, "height": 56,
                "fill_color": "rgba(232,106,58,1)", "group_id": "tag",
            },
            {
                **_text("tag_label", "SMALL BATCH", 20, 96, w=210, h=28),
                "x": 575, "group_id": "tag",
            },
            {
                "id": "flag_bg", "type": "shape", "shape_type": "rectangle",
                "x": 680, "y": 70, "width": 170, "height": 36,
                "fill_color": "rgba(180,40,40,1)", "rotation": -18,
                "group_id": "flag",
            },
            {
                **_text("flag_label", "OPEN LATE", 16, 78, w=140, h=20),
                "x": 695, "rotation": -18, "group_id": "flag",
            },
        ], w=900, h=900)
        for target in ({"width": 960, "height": 240}, {"width": 640, "height": 360}):
            output = resize(source, target)
            self.assertEqual(
                [element["id"] for element in output["elements"]],
                [element["id"] for element in source["elements"]],
            )
            flag = next(element for element in output["elements"] if element["id"] == "flag_label")
            self.assertEqual(flag["content"], "OPEN LATE")
            self.assertEqual(flag["font_family"], "Solstice Sans")
            self.assertTrue(self._inside(flag, target), target)
            flag_bg = next(element for element in output["elements"] if element["id"] == "flag_bg")
            self.assertTrue(self._inside(flag_bg, target), target)
            if target["height"] <= 240:
                # Independent x/y scale would leave this accent about 10 px tall.
                self.assertGreaterEqual(flag_bg["height"], 16.0)
                self.assertLess(flag_bg["width"] / flag_bg["height"], 8.0)

    def test_a_full_pill_label_unwraps_when_one_line_fits_the_shape(self):
        from engine.resize import _fit_single_label_to_shape, _measured_lines

        shape = {
            "id": "pill", "type": "shape", "x": 0, "y": 0, "width": 220, "height": 48,
            "shape_type": "rectangle",
        }
        text = {
            "id": "label", "type": "text", "x": 16, "y": 4, "width": 188, "height": 40,
            "content": "LIMITED EDITION", "font_family": "Solstice Sans",
            "font_size": 24, "font_weight": "bold", "color": "rgba(255,255,255,1)",
        }
        before = _measured_lines(text["content"], text["font_family"], text["font_weight"], None, text["font_size"], text["width"])
        self.assertGreater(before[0], 1)
        _fit_single_label_to_shape([shape, text], 8)
        after = _measured_lines(text["content"], text["font_family"], text["font_weight"], None, text["font_size"], text["width"])
        self.assertEqual(after[0], 1)
        self.assertEqual(text["font_size"], 24)

    def test_level2_circle_stays_round_and_on_canvas(self):
        from engine.contract import contract_failures

        task = json.loads((ROOT / "input" / "chameleon_sample_level2_core.json").read_text())
        output = resize(task["source"], task["target_canvas"])
        circle = next(element for element in output["elements"] if element["id"] == "deco_circle_1")
        self.assertEqual(circle["width"], circle["height"])
        self.assertEqual(
            contract_failures(task["source"], output, task["target_canvas"], ROOT / "brand_kit" / "assets"),
            [],
        )

    def test_a_source_circle_past_the_edge_is_scaled_onto_a_short_canvas(self):
        source = _design([
            {
                "id": "blob", "type": "shape", "shape_type": "ellipse",
                "x": -80, "y": -40, "width": 360, "height": 360,
                "fill_color": "rgba(255,255,255,0.2)",
            },
            _text("title", "Night market", 40, 80, w=400, h=60),
        ], w=800, h=800)
        for target in ({"width": 900, "height": 200}, {"width": 240, "height": 640}):
            output = resize(source, target)
            blob = next(element for element in output["elements"] if element["id"] == "blob")
            self.assertAlmostEqual(blob["width"], blob["height"], places=3)
            self.assertGreaterEqual(blob["x"], -0.5)
            self.assertGreaterEqual(blob["y"], -0.5)
            self.assertLessEqual(blob["x"] + blob["width"], target["width"] + 0.5)
            self.assertLessEqual(blob["y"] + blob["height"], target["height"] + 0.5)
            title = next(element for element in output["elements"] if element["id"] == "title")
            self.assertEqual(title["content"], "Night market")

    def test_judged_strip_keeps_the_content_image_visible(self):
        task = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())
        source = task["source"]
        texts = {item["id"]: item["content"] for item in source["elements"] if item["type"] == "text"}
        judgments = {
            eid: _judgment(content, "body", "medium", False) for eid, content in texts.items()
        }
        judgments["headline"] = _judgment(texts["headline"], "headline", "primary", True)
        judgments["cta_label"] = _judgment(texts["cta_label"], "call_to_action", "high", False)
        judgments["legal_text"] = _judgment(texts["legal_text"], "disclosure", "low", True)
        output = resize(source, {"width": 728, "height": 90}, judgments=judgments)
        product = next(element for element in output["elements"] if element["id"] == "product")
        headline = next(element for element in output["elements"] if element["id"] == "headline")
        action = next(element for element in output["elements"] if element["id"] == "cta_label")
        self.assertGreaterEqual(min(product["width"], product["height"]), 28.0)
        self.assertAlmostEqual(product["width"] / product["height"], 1.0, places=2)
        self.assertGreaterEqual(headline["font_size"], 8.0)
        self.assertEqual(action["content"], texts["cta_label"])
        self.assertEqual(product["asset_id"], "solstice_bottle.png")


class ClassifiedPillAndOfferTests(unittest.TestCase):
    def test_logo_park_uses_classified_pill_labels_not_the_element_id(self):
        from engine.resize import _parked_logo_clears_the_headline

        def line(eid, size, x, y, width=90):
            return {
                "id": eid, "type": "text", "content": "Mark", "font_family": "Solstice Sans",
                "font_size": size, "font_weight": "bold", "text_align": "left",
                "x": x, "y": y, "width": width, "height": 16,
            }

        current = [
            line("headline", 10, 8, 8, 300),
            line("mark_copy", 8, 8, 50),
            line("deco_badge_note", 8, 400, 50),
            line("shop_note", 11, 520, 8, 120),
        ]
        parked = [
            line("headline", 13, 8, 8, 380),
            line("mark_copy", 8, 8, 50),
            line("deco_badge_note", 6, 400, 50),
            line("shop_note", 11, 520, 8, 120),
            {"id": "logo", "type": "image", "x": 8, "y": 80, "width": 70, "height": 18, "asset_id": "logo.png"},
        ]
        # deco_badge_note has "badge" in its id and got smaller. It is not a
        # classified pill, so it must not decide the park. mark_copy has no
        # such word and is the pill label that has to stay.
        self.assertTrue(_parked_logo_clears_the_headline(
            current, parked, "headline", None, "shop_note", "logo", None, {"mark_copy"}, 728, 90,
        ))
        shrunk = [dict(element) for element in parked]
        shrunk[1] = line("mark_copy", 6, 8, 50)
        self.assertFalse(_parked_logo_clears_the_headline(
            current, shrunk, "headline", None, "shop_note", "logo", None, {"mark_copy"}, 728, 90,
        ))
        # A 50 px gain is real on a 728 px banner and a sliver when the logo
        # that left the row was 400 px wide. The same numbers on a 40 px
        # headline are not an 8% step.
        wide_logo = list(current) + [
            {"id": "logo", "type": "image", "x": 400, "y": 8, "width": 400, "height": 24, "asset_id": "logo.png"},
        ]
        self.assertFalse(_parked_logo_clears_the_headline(
            wide_logo, parked, "headline", None, "shop_note", "logo", None, {"mark_copy"}, 1600, 100,
        ))
        tall = [dict(element) for element in parked]
        tall[0] = line("headline", 41.4, 8, 8, 380)
        current_tall = [line("headline", 40, 8, 8, 300)] + current[1:]
        self.assertFalse(_parked_logo_clears_the_headline(
            current_tall, tall, "headline", None, "shop_note", "logo", None, {"mark_copy"}, 728, 90,
        ))
        # 15 px is still 16% of a 90 px strip, so the height floor passes.
        # It is just over half of the 28 px logo that already fit the row,
        # and that drop is the rejection.
        halved = [dict(element) for element in parked]
        halved[-1] = dict(halved[-1], height=15)
        current_with_logo = list(current) + [
            {"id": "logo", "type": "image", "x": 8, "y": 8, "width": 80, "height": 28, "asset_id": "logo.png"},
        ]
        self.assertFalse(_parked_logo_clears_the_headline(
            current_with_logo, halved, "headline", None, "shop_note", "logo", None, {"mark_copy"}, 728, 90,
        ))

    def test_a_foreground_shape_blocks_the_offer_and_a_backdrop_does_not(self):
        from engine.resize import _relocate_offer_ribbons

        def design(extra=None):
            elements = [
                {"id": "price", "type": "text", "x": 28, "y": 72, "width": 200, "height": 80,
                 "content": "$39", "font_family": "Solstice Sans", "font_size": 48, "font_weight": "bold",
                 "text_align": "left"},
                {"id": "plate", "type": "shape", "shape_type": "rectangle", "x": 900, "y": 28,
                 "width": 180, "height": 40, "fill_color": "rgba(180,40,40,1)", "group_id": "g", "rotation": -12},
                {"id": "offer", "type": "text", "x": 900, "y": 32, "width": 180, "height": 30,
                 "content": "50% OFF", "font_family": "Solstice Sans", "font_size": 16, "font_weight": "bold",
                 "text_align": "center", "group_id": "g", "rotation": -12},
            ]
            if extra is not None:
                elements.append(extra)
            return {"canvas": {"width": 1200, "height": 628}, "elements": elements}

        panel = {
            "id": "panel", "type": "shape", "shape_type": "rectangle",
            "x": 480, "y": 8, "width": 700, "height": 420, "fill_color": "rgba(20,20,80,1)", "opacity": 1,
        }
        backdrop = {
            "id": "ground", "type": "shape", "shape_type": "rectangle",
            "x": 0, "y": 0, "width": 1200, "height": 628, "fill_color": "rgba(240,240,240,1)", "opacity": 1,
        }
        blocked = _relocate_offer_ribbons(design(panel), {"offer"}, 1200, 628)
        open_move = _relocate_offer_ribbons(design(backdrop), {"offer"}, 1200, 628)
        blocked_font = next(element["font_size"] for element in blocked["elements"] if element["id"] == "offer")
        open_font = next(element["font_size"] for element in open_move["elements"] if element["id"] == "offer")
        self.assertLess(blocked_font, 20)
        self.assertGreater(open_font, 40)

    def test_sale_card_keeps_the_baseline_text_and_records_the_offer_move(self):
        source = json.loads((ROOT / "eval" / "text_context" / "designs" / "d08.json").read_text())
        texts = {element["id"]: element["content"] for element in source["elements"] if element["type"] == "text"}
        judgments = {
            eid: _judgment(content, "body", "medium", False) for eid, content in texts.items()
        }
        judgments["t01"] = _judgment(texts["t01"], "discount", "high", True)
        judgments["t04"] = _judgment(texts["t04"], "price", "high", True)
        judgments["t02"] = _judgment(texts["t02"], "headline", "primary", True)
        output = resize(source, {"width": 1200, "height": 628}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(layout_bridge.last_decision, "baseline_kept")
        self.assertTrue(layout_bridge.last_offer_moved)
        self.assertGreater(by_id["t01"]["font_size"], 40)
        self.assertGreater(by_id["t04"]["font_size"], by_id["t02"]["font_size"])
        self.assertEqual(by_id["t01"]["content"], "50% OFF")
        self.assertEqual(by_id["a01"]["fill_color"], "rgba(180,40,40,1)")

    def test_renamed_copy_follows_the_canvas_shape_and_a_nearby_shape(self):
        from copy import deepcopy
        from engine.resize import _axis_overlap, _visual_bbox

        def rewrite(design, ids, groups, words):
            copied = deepcopy(design)
            for element in copied["elements"]:
                element["id"] = ids.get(element["id"], element["id"])
                if element.get("group_id"):
                    element["group_id"] = groups.get(element["group_id"], element["group_id"])
                if element.get("type") == "text" and element["id"] in words:
                    element["content"] = words[element["id"]]
            return copied

        banner_source = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())["source"]
        banner = rewrite(
            banner_source,
            {
                "bg": "ground", "logo": "mark", "product": "photo", "headline": "lead",
                "subhead": "aside", "badge1_bg": "chip_a_shape", "badge1_label": "chip_a_words",
                "badge2_bg": "chip_b_shape", "badge2_label": "chip_b_words",
                "badge3_bg": "chip_c_shape", "badge3_label": "chip_c_words",
                "cta_bg": "action_shape", "cta_label": "action_words",
                "ribbon_bg": "tilt_shape", "ribbon_label": "tilt_words",
                "holiday_watermark": "stamp", "legal_text": "fine",
            },
            {"badge1": "chip_a", "badge2": "chip_b", "badge3": "chip_c", "cta": "action", "ribbon": "tilt"},
            {
                "lead": "North wind market opens at dawn",
                "aside": "Wool blankets, cedar soap, and a short walk from the ferry.",
                "chip_a_words": "FIRST PRESS", "chip_b_words": "HARBOR LOT", "chip_c_words": "SMALL RUN",
                "action_words": "Reserve a crate", "tilt_words": "THIS WEEK",
                "fine": "Sold by the crate. Ask at the stall for the full list of weights.",
            },
        )
        texts = {element["id"]: element["content"] for element in banner["elements"] if element["type"] == "text"}
        banner_judgments = {eid: _judgment(content, "body", "medium", False) for eid, content in texts.items()}
        banner_judgments["lead"] = _judgment(texts["lead"], "headline", "primary", True)
        banner_judgments["action_words"] = _judgment(texts["action_words"], "call_to_action", "high", False)
        banner_judgments["fine"] = _judgment(texts["fine"], "disclosure", "low", True)
        wide_banner = resize(banner, {"width": 1000, "height": 120}, judgments=banner_judgments)
        short_card = resize(banner, {"width": 640, "height": 160}, judgments=banner_judgments)
        wide_by = {element["id"]: element for element in wide_banner["elements"]}
        short_by = {element["id"]: element for element in short_card["elements"]}
        # The logo has to stay a large share of this short side. A park that
        # cuts it in half to buy headline width is rejected. The product is a
        # column, and the headline stays larger than the chips.
        self.assertGreaterEqual(wide_by["mark"]["height"], 48.0)
        self.assertGreaterEqual(wide_by["photo"]["height"], 40.0)
        self.assertGreater(wide_by["lead"]["font_size"], wide_by["chip_a_words"]["font_size"])
        self.assertGreaterEqual(wide_by["fine"]["font_size"], 8.0)
        self.assertGreaterEqual(wide_by["action_words"]["font_size"], 16.0)
        self.assertGreater(short_by["lead"]["font_size"], short_by["chip_a_words"]["font_size"])
        self.assertGreater(short_by["lead"]["font_size"], short_by["chip_b_words"]["font_size"])
        self.assertGreater(short_by["lead"]["font_size"], short_by["chip_c_words"]["font_size"])
        self.assertGreaterEqual(short_by["mark"]["height"], 30.0)
        self.assertGreaterEqual(short_by["photo"]["height"], 60.0)
        self.assertEqual(wide_by["lead"]["content"], "North wind market opens at dawn")
        tall_banner = resize(banner, {"width": 1456, "height": 180}, judgments=banner_judgments)
        tall_by = {element["id"]: element for element in tall_banner["elements"]}
        self.assertGreater(tall_by["lead"]["font_size"], tall_by["chip_a_words"]["font_size"])
        self.assertGreater(tall_by["lead"]["font_size"], 30.0)
        self.assertGreaterEqual(tall_by["fine"]["font_size"], 7.5)

        sale_source = json.loads((ROOT / "eval" / "text_context" / "designs" / "d08.json").read_text())
        sale = rewrite(
            sale_source,
            {"t04": "cost", "t02": "title", "a01": "plate", "t01": "offer", "t03": "when", "t05": "note"},
            {"g01": "cluster"},
            {"cost": "$18", "title": "Harbor coat", "offer": "Two for one", "when": "This week", "note": "Wool, charcoal."},
        )
        sale_texts = {element["id"]: element["content"] for element in sale["elements"] if element["type"] == "text"}
        sale_judgments = {eid: _judgment(content, "body", "medium", False) for eid, content in sale_texts.items()}
        sale_judgments["offer"] = _judgment(sale_texts["offer"], "discount", "high", True)
        sale_judgments["cost"] = _judgment(sale_texts["cost"], "price", "high", True)
        sale_judgments["title"] = _judgment(sale_texts["title"], "headline", "primary", True)
        card = resize(sale, {"width": 900, "height": 480}, judgments=sale_judgments)
        self.assertTrue(layout_bridge.last_offer_moved)
        banner_out = resize(sale, {"width": 1600, "height": 220}, judgments=sale_judgments)
        self.assertTrue(layout_bridge.last_offer_moved)
        smaller = resize(sale, {"width": 600, "height": 314}, judgments=sale_judgments)
        self.assertTrue(layout_bridge.last_offer_moved)
        blocked = deepcopy(sale)
        blocked["elements"].append({
            "id": "panel", "type": "shape", "shape_type": "rectangle",
            "x": 560, "y": 40, "width": 460, "height": 520,
            "fill_color": "rgba(20,40,90,1)", "opacity": 1,
        })
        beside = resize(blocked, {"width": 900, "height": 480}, judgments=sale_judgments)
        card_by = {element["id"]: element for element in card["elements"]}
        banner_by = {element["id"]: element for element in banner_out["elements"]}
        small_by = {element["id"]: element for element in smaller["elements"]}
        beside_by = {element["id"]: element for element in beside["elements"]}
        self.assertGreater(card_by["cost"]["font_size"], card_by["title"]["font_size"])
        self.assertGreater(card_by["offer"]["font_size"], 30.0)
        self.assertGreaterEqual(banner_by["offer"]["font_size"], 24.0)
        self.assertGreaterEqual(banner_by["cost"]["font_size"], 18.0)
        self.assertGreaterEqual(banner_by["title"]["font_size"], 18.0)
        self.assertLess(banner_by["cost"]["width"], 800.0)
        self.assertGreater(banner_by["offer"]["x"], 560.0)
        self.assertGreater(small_by["cost"]["font_size"], small_by["title"]["font_size"])
        panel = (beside_by["panel"]["x"], beside_by["panel"]["y"], beside_by["panel"]["width"], beside_by["panel"]["height"])
        ribbon = _visual_bbox([beside_by["plate"], beside_by["offer"]])
        self.assertFalse(_axis_overlap(ribbon, panel, 0))
        self.assertEqual(beside_by["offer"]["content"], "Two for one")
        self.assertGreater(beside_by["cost"]["font_size"], beside_by["title"]["font_size"])

    def test_empty_headline_and_legal_slots_keep_the_clearer_picture(self):
        from copy import deepcopy

        source = json.loads((ROOT / "input" / "chameleon_sample_level3_advanced.json").read_text())["source"]
        banner = deepcopy(source)
        words = {
            "headline": "North wind market opens at dawn",
            "subhead": "Wool blankets, cedar soap, and a short walk from the ferry.",
            "badge1_label": "FIRST PRESS", "badge2_label": "HARBOR LOT", "badge3_label": "SMALL RUN",
            "cta_label": "Reserve a crate", "ribbon_label": "THIS WEEK",
            "legal_text": "Sold by the crate. Ask at the stall for the full list of weights.",
        }
        for element in banner["elements"]:
            if element.get("type") == "text" and element["id"] in words:
                element["content"] = words[element["id"]]
        texts = {element["id"]: element["content"] for element in banner["elements"] if element["type"] == "text"}
        judgments = {eid: _judgment(content, "body", "high", True) for eid, content in texts.items()}
        judgments["legal_text"] = _judgment(texts["legal_text"], "badge", "low", False)
        judgments["ribbon_label"] = _judgment(texts["ribbon_label"], "headline", "primary", True)
        judgments["cta_label"] = _judgment(texts["cta_label"], "call_to_action", "high", True)
        for label in ("badge1_label", "badge2_label", "badge3_label"):
            judgments[label] = _judgment(texts[label], "badge", "medium", False)
        original = json.loads(json.dumps(judgments))
        canvas = {"width": 1000, "height": 120}
        output = resize(banner, canvas, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(judgments, original)
        override = layout_bridge.last_layout_override
        self.assertIsNotNone(override)
        self.assertEqual(override["kind"], "structural_slots")
        self.assertEqual(override["model_role"]["headline"], "body")
        self.assertEqual(override["model_role"]["legal_text"], "badge")
        self.assertGreaterEqual(by_id["product"]["height"], 40.0)
        self.assertGreaterEqual(by_id["legal_text"]["font_size"], 8.0)
        self.assertGreater(by_id["headline"]["font_size"], by_id["badge1_label"]["font_size"])
        self.assertEqual(layout_bridge.last_decision, "judged")

    def test_a_grouped_panel_headline_is_not_replaced_when_the_image_would_shrink(self):
        design = _design([
            {"id": "ground", "type": "shape", "shape_type": "rectangle", "x": 0, "y": 0,
             "width": 1200, "height": 900, "fill_color": "#222244"},
            {"id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "fit": "contain",
             "x": 760, "y": 180, "width": 360, "height": 360},
            {"id": "panel_shape", "type": "shape", "shape_type": "rectangle", "group_id": "panel",
             "x": 60, "y": 180, "width": 640, "height": 180, "fill_color": "#ffffff"},
            {"id": "panel_words", "type": "text", "group_id": "panel", "content": "Harbour market",
             "font_size": 72, "font_family": "Solstice Sans", "color": "#111111",
             "x": 80, "y": 220, "width": 600, "height": 100},
            _text("pier", "Open stalls along the pier", 28, 420, w=640, h=48),
            _text("lot", "Lot card", 16, 500, w=200, h=28),
            _text("fine", "Weights are posted beside the stall.", 9, 840, w=700, h=22),
            {"id": "corner_shape", "type": "shape", "shape_type": "rectangle", "group_id": "corner",
             "rotation": -12, "x": 980, "y": 30, "width": 180, "height": 48, "fill_color": "#aa2222"},
            {"id": "corner_words", "type": "text", "group_id": "corner", "rotation": -12, "content": "TODAY",
             "font_size": 16, "font_family": "Solstice Sans", "color": "#ffffff",
             "x": 1000, "y": 40, "width": 140, "height": 28},
        ], w=1200, h=900)
        texts = {element["id"]: element["content"] for element in design["elements"] if element["type"] == "text"}
        judgments = {eid: _judgment(content, "badge", "low", False) for eid, content in texts.items()}
        judgments["panel_words"] = _judgment(texts["panel_words"], "headline", "primary", True)
        judgments["pier"] = _judgment(texts["pier"], "body", "medium", False)
        output = resize(design, {"width": 1080, "height": 1080}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertIsNone(layout_bridge.last_layout_override)
        self.assertEqual(judgments["panel_words"]["role"], "headline")
        self.assertGreater(by_id["panel_words"]["font_size"], by_id["corner_words"]["font_size"])
        self.assertGreater(by_id["photo"]["height"], 150.0)


class RotatedActionCandidateTests(unittest.TestCase):
    def _sale(self, words):
        from copy import deepcopy

        source = json.loads((ROOT / "eval" / "text_context" / "designs" / "d08.json").read_text())
        copied = deepcopy(source)
        ids = {"t04": "cost", "t02": "title", "a01": "plate", "t01": "offer", "t03": "when", "t05": "note"}
        for element in copied["elements"]:
            element["id"] = ids.get(element["id"], element["id"])
            if element.get("group_id"):
                element["group_id"] = "cluster"
            if element.get("type") == "text" and element["id"] in words:
                element["content"] = words[element["id"]]
        return copied

    def _judge(self, design, roles):
        judgments = {}
        for element in design["elements"]:
            if element.get("type") != "text":
                continue
            role, prominence = roles[element["id"]]
            judgments[element["id"]] = _judgment(element["content"], role, prominence, role in {"price", "call_to_action"})
        return judgments

    def test_the_only_high_rotated_action_grows_without_a_new_role(self):
        design = self._sale({
            "cost": "$18", "title": "Harbor coat", "offer": "Two for one",
            "when": "This week", "note": "Wool, charcoal.",
        })
        judgments = self._judge(design, {
            "offer": ("call_to_action", "high"),
            "cost": ("price", "high"),
            "title": ("headline", "primary"),
            "when": ("body", "medium"),
            "note": ("other", "low"),
        })
        output = resize(design, {"width": 1600, "height": 220}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(judgments["offer"]["role"], "call_to_action")
        self.assertFalse(layout_bridge.last_offer_moved)
        self.assertTrue(layout_bridge.last_accent_moved)
        self.assertGreaterEqual(by_id["offer"]["font_size"], 24.0)
        self.assertGreater(by_id["cost"]["font_size"], by_id["title"]["font_size"])

    def test_a_second_action_keeps_the_rotated_label_small(self):
        design = self._sale({
            "cost": "$18", "title": "Harbor coat", "offer": "Two for one",
            "when": "This week", "note": "Wool, charcoal.",
        })
        design["elements"].extend([
            {"id": "btn_shape", "type": "shape", "shape_type": "rectangle", "group_id": "btn",
             "x": 400, "y": 520, "width": 280, "height": 72, "fill_color": "rgba(20,90,40,1)"},
            {"id": "btn_words", "type": "text", "group_id": "btn", "content": "Hold a place",
             "font_size": 22, "font_family": "Solstice Sans", "color": "#ffffff",
             "x": 420, "y": 540, "width": 240, "height": 32},
        ])
        judgments = self._judge(design, {
            "offer": ("call_to_action", "high"),
            "btn_words": ("call_to_action", "high"),
            "cost": ("price", "high"),
            "title": ("headline", "primary"),
            "when": ("body", "medium"),
            "note": ("other", "low"),
        })
        output = resize(design, {"width": 1600, "height": 220}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertFalse(layout_bridge.last_accent_moved)
        self.assertFalse(layout_bridge.last_offer_moved)
        self.assertLess(by_id["offer"]["font_size"], 16.0)
        self.assertGreater(by_id["btn_words"]["font_size"], by_id["offer"]["font_size"])

    def test_a_high_status_accent_is_not_treated_as_the_action(self):
        design = self._sale({
            "cost": "$18", "title": "Harbor coat", "offer": "On the shelf",
            "when": "This week", "note": "Wool, charcoal.",
        })
        design["elements"].extend([
            {"id": "btn_shape", "type": "shape", "shape_type": "rectangle", "group_id": "btn",
             "x": 400, "y": 520, "width": 280, "height": 72, "fill_color": "rgba(20,90,40,1)"},
            {"id": "btn_words", "type": "text", "group_id": "btn", "content": "Ask at the desk",
             "font_size": 22, "font_family": "Solstice Sans", "color": "#ffffff",
             "x": 420, "y": 540, "width": 240, "height": 32},
        ])
        judgments = self._judge(design, {
            "offer": ("badge", "high"),
            "btn_words": ("call_to_action", "high"),
            "cost": ("price", "high"),
            "title": ("headline", "primary"),
            "when": ("body", "medium"),
            "note": ("other", "low"),
        })
        output = resize(design, {"width": 1600, "height": 220}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(judgments["offer"]["role"], "badge")
        self.assertFalse(layout_bridge.last_accent_moved)
        self.assertLess(by_id["offer"]["font_size"], 16.0)
        self.assertGreater(by_id["btn_words"]["font_size"], by_id["offer"]["font_size"])

    def test_a_restored_reading_line_stays_when_the_only_action_grows(self):
        design = {
            "canvas": {"width": 1000, "height": 800},
            "elements": [
                {"id": "cost", "type": "text", "content": "$9", "font_family": "Solstice Sans",
                 "font_size": 42, "font_weight": "bold", "x": 60, "y": 80, "width": 180, "height": 56,
                 "color": "rgba(20,20,20,1)", "text_align": "left"},
                {"id": "title", "type": "text", "content": "Cedar crates", "font_family": "Solstice Serif",
                 "font_size": 32, "font_weight": "bold", "x": 60, "y": 180, "width": 520, "height": 48,
                 "color": "rgba(20,20,20,1)", "text_align": "left"},
                {"id": "note", "type": "text", "content": "Cut this morning.", "font_family": "Solstice Sans",
                 "font_size": 16, "x": 60, "y": 260, "width": 360, "height": 28,
                 "color": "rgba(20,20,20,1)", "text_align": "left"},
                {"id": "plate", "type": "shape", "shape_type": "rectangle", "group_id": "tilt",
                 "rotation": -14, "x": 760, "y": 40, "width": 200, "height": 48,
                 "fill_color": "rgba(160,40,40,1)"},
                {"id": "mark", "type": "text", "content": "Step this way", "font_family": "Solstice Sans",
                 "font_size": 22, "font_weight": "bold", "group_id": "tilt", "rotation": -14,
                 "x": 780, "y": 50, "width": 160, "height": 28, "color": "rgba(255,255,255,1)"},
            ],
        }
        judgments = self._judge(design, {
            "mark": ("call_to_action", "high"),
            "cost": ("price", "high"),
            "title": ("product_name", "medium"),
            "note": ("body", "low"),
        })
        output = resize(design, {"width": 1400, "height": 360}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(judgments["mark"]["role"], "call_to_action")
        self.assertTrue(layout_bridge.last_accent_moved)
        self.assertFalse(layout_bridge.last_offer_moved)
        self.assertIsNotNone(layout_bridge.last_layout_override)
        self.assertGreaterEqual(by_id["mark"]["font_size"], 24.0)
        self.assertGreater(by_id["cost"]["font_size"], by_id["mark"]["font_size"])


def _row(role, prominence, uncertainty="low"):
    return {"role": role, "prominence": prominence, "uncertainty": uncertainty, "must_read": True}


class ProminenceRankTests(unittest.TestCase):
    def test_the_more_prominent_headline_wins_in_either_order(self):
        from engine.text_context.layout_bridge import _pick_headline

        quiet = {"id": "quiet", "type": "text", "content": "A side note", "font_size": 18, "y": 10}
        loud = {"id": "loud", "type": "text", "content": "The main line", "font_size": 48, "y": 80}
        by_id = {item["id"]: item for item in (quiet, loud)}
        usable = {"quiet": _row("headline", "minimal"), "loud": _row("headline", "high")}
        for order in ((quiet, loud), (loud, quiet)):
            chosen = _pick_headline(usable, list(order), by_id, quiet)
            self.assertEqual(chosen["id"], "loud")

    def test_equal_headline_prominence_keeps_the_earlier_candidate(self):
        from engine.text_context.layout_bridge import _pick_headline

        first = {"id": "first", "type": "text", "content": "First line", "font_size": 30, "y": 10}
        second = {"id": "second", "type": "text", "content": "Second line", "font_size": 30, "y": 80}
        by_id = {item["id"]: item for item in (first, second)}
        usable = {"first": _row("headline", "medium"), "second": _row("headline", "medium")}
        self.assertEqual(_pick_headline(usable, [first, second], by_id, first)["id"], "first")
        self.assertEqual(_pick_headline(usable, [second, first], by_id, first)["id"], "second")

    def test_low_uncertainty_breaks_an_equal_headline_tie(self):
        from engine.text_context.layout_bridge import _pick_headline

        shaky = {"id": "shaky", "type": "text", "content": "Maybe this", "font_size": 30, "y": 10}
        sure = {"id": "sure", "type": "text", "content": "This one", "font_size": 30, "y": 80}
        by_id = {item["id"]: item for item in (shaky, sure)}
        usable = {
            "shaky": _row("headline", "high", uncertainty="high"),
            "sure": _row("headline", "high", uncertainty="low"),
        }
        for order in ((shaky, sure), (sure, shaky)):
            self.assertEqual(_pick_headline(usable, list(order), by_id, shaky)["id"], "sure")

    def test_the_more_prominent_loose_action_wins_in_either_order(self):
        from engine.text_context.layout_bridge import _pick_loose_cta

        quiet = {"id": "quiet", "type": "text", "content": "Learn more", "font_size": 16, "y": 10}
        loud = {"id": "loud", "type": "text", "content": "Book now", "font_size": 28, "y": 80}
        usable = {"quiet": _row("call_to_action", "minimal"), "loud": _row("call_to_action", "primary")}
        for order in ((quiet, loud), (loud, quiet)):
            chosen = _pick_loose_cta(usable, list(order), None, None)
            self.assertEqual(chosen["id"], "loud")

    def test_the_more_prominent_grouped_action_wins_in_either_order(self):
        from engine.text_context.layout_bridge import _pick_cta

        def pill(gid, label, width):
            return (gid, [
                {"id": gid + "_bg", "type": "shape", "x": 0, "y": 0, "width": width, "height": 40},
                {"id": label, "type": "text", "x": 0, "y": 0, "width": width, "height": 20, "content": label},
            ])

        quiet = pill("quiet", "details", 220)
        loud = pill("loud", "tickets", 80)
        usable = {"details": _row("call_to_action", "minimal"), "tickets": _row("call_to_action", "high")}
        for order in ((quiet, loud), (loud, quiet)):
            chosen, _badges, loose = _pick_cta(usable, list(order), None, [], None)
            self.assertIsNone(loose)
            self.assertEqual(chosen[0], "loud")

    def test_equal_grouped_actions_keep_the_earlier_pill(self):
        from engine.text_context.layout_bridge import _pick_cta

        def pill(gid):
            return (gid, [
                {"id": gid + "_bg", "type": "shape", "x": 0, "y": 0, "width": 100, "height": 40},
                {"id": gid + "_label", "type": "text", "x": 0, "y": 0, "width": 80, "height": 20, "content": gid},
            ])

        left, right = pill("left"), pill("right")
        usable = {
            "left_label": _row("call_to_action", "high"),
            "right_label": _row("call_to_action", "high"),
        }
        chosen, _badges, _loose = _pick_cta(usable, [left, right], None, [], None)
        self.assertEqual(chosen[0], "left")
        chosen, _badges, _loose = _pick_cta(usable, [right, left], None, [], None)
        self.assertEqual(chosen[0], "right")

    def test_legal_selection_still_prefers_a_must_read_disclosure(self):
        from engine.text_context.layout_bridge import _pick_legal

        optional = {"id": "optional", "type": "text", "content": "See terms", "font_size": 14, "y": 10}
        required = {"id": "required", "type": "text", "content": "Void where prohibited", "font_size": 12, "y": 40}
        by_id = {item["id"]: item for item in (optional, required)}
        usable = {
            "optional": {"role": "disclosure", "prominence": "primary", "uncertainty": "low", "must_read": False},
            "required": {"role": "disclosure", "prominence": "low", "uncertainty": "low", "must_read": True},
        }
        for order in ((optional, required), (required, optional)):
            chosen = _pick_legal(usable, list(order), by_id, None, optional)
            self.assertEqual(chosen["id"], "required")


class BadgeCapTests(unittest.TestCase):
    def test_short_badge_keeps_its_proportions_on_a_portrait_resize(self):
        from engine.resize import _text_overflow_ids

        source = json.loads((ROOT / "input" / "chameleon_sample_level2_core.json").read_text())["source"]
        judgments = {
            element["id"]: _judgment(
                element["content"],
                {"headline": "headline", "subhead": "subheadline", "badge_label": "badge",
                 "cta_label": "call_to_action"}[element["id"]],
                {"headline": "primary", "subhead": "medium", "badge_label": "low",
                 "cta_label": "high"}[element["id"]],
                element["id"] == "cta_label",
            )
            for element in source["elements"] if element["type"] == "text"
        }
        output = resize(source, {"width": 1080, "height": 1350}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        source_badge = next(element for element in source["elements"] if element["id"] == "badge_bg")
        badge = by_id["badge_bg"]

        self.assertEqual(layout_bridge.last_decision, "judged")
        self.assertLessEqual(
            badge["width"] / badge["height"],
            1.2 * source_badge["width"] / source_badge["height"],
        )
        self.assertNotIn("badge_label", _text_overflow_ids(output))
        self.assertGreater(by_id["cta_label"]["font_size"], by_id["badge_label"]["font_size"])

        # The local model has also called this grouped accent a headline.
        # Its badge slot should still yield to a larger readable action.
        judgments["badge_label"] = _judgment(
            next(element["content"] for element in source["elements"] if element["id"] == "badge_label"),
            "headline", "high", True,
        )
        output = resize(source, {"width": 1080, "height": 1350}, judgments=judgments)
        by_id = {element["id"]: element for element in output["elements"]}
        self.assertEqual(layout_bridge.last_decision, "judged")
        self.assertLessEqual(
            by_id["badge_bg"]["width"] / by_id["badge_bg"]["height"],
            1.2 * source_badge["width"] / source_badge["height"],
        )
        self.assertNotIn("badge_label", _text_overflow_ids(output))
        self.assertGreater(by_id["cta_label"]["font_size"], by_id["badge_label"]["font_size"])

    def test_ordinary_badges_are_capped_at_medium_and_quiet_ones_stay_quiet(self):
        from engine.text_context.layout_bridge import PROMINENCE_IMPORTANCE, _importance_by_id

        expected = {
            "primary": "medium",
            "high": "medium",
            "medium": "medium",
            "low": "low",
            "minimal": "minimal",
        }
        for prominence, capped in expected.items():
            weights = _importance_by_id(
                {"mark": {"role": "badge", "prominence": prominence}},
                {"mark": "badge"},
            )
            self.assertEqual(weights["mark"], PROMINENCE_IMPORTANCE[capped], prominence)

    def test_a_discount_badge_keeps_the_model_prominence(self):
        from engine.text_context.layout_bridge import PROMINENCE_IMPORTANCE, _importance_by_id

        for prominence in ("primary", "high", "medium", "low", "minimal"):
            weights = _importance_by_id(
                {"offer": {"role": "discount", "prominence": prominence}},
                {"offer": "badge"},
            )
            self.assertEqual(weights["offer"], PROMINENCE_IMPORTANCE[prominence], prominence)

    def test_non_badge_text_and_special_slots_keep_their_rules(self):
        from engine.text_context.layout_bridge import PROMINENCE_IMPORTANCE, _importance_by_id

        weights = _importance_by_id(
            {
                "title": {"role": "headline", "prominence": "primary"},
                "extra": {"role": "call_to_action", "prominence": "primary"},
                "mistaken": {"role": "headline", "prominence": "primary"},
            },
            {"title": "headline", "extra": "badge", "mistaken": "badge"},
        )
        self.assertEqual(weights["title"], PROMINENCE_IMPORTANCE["primary"])
        self.assertEqual(weights["extra"], PROMINENCE_IMPORTANCE["low"])
        self.assertEqual(weights["mistaken"], PROMINENCE_IMPORTANCE["minimal"])


class ScorerTests(unittest.TestCase):
    def test_a_label_on_its_own_shape_is_not_an_overlap_penalty(self):
        from engine.resize import _score_candidate

        shape = {
            "id": "btn", "type": "shape", "group_id": "action",
            "x": 10, "y": 10, "width": 200, "height": 50,
        }
        inside = {
            "id": "label", "type": "text", "group_id": "action",
            "x": 30, "y": 20, "width": 150, "height": 30,
            "content": "Come in", "font_family": "Solstice Sans", "font_size": 18,
        }
        outside = dict(inside, x=250, y=80)
        weights = {"btn": 1.0, "label": 1.0}
        grouped = _score_candidate([shape, inside], weights, 800, 400)
        separated = _score_candidate([shape, outside], weights, 800, 400)
        self.assertEqual(grouped, 0.0)
        self.assertLessEqual(grouped, separated)

    def test_unbreakable_ink_wider_than_the_box_is_penalized(self):
        from engine.resize import _score_candidate

        token = "https://example.com/stores/VeryLongSolsticeStoreName"
        weights = {"url": 1.0}
        for width in (40, 280):
            element = {
                "id": "url", "type": "text", "x": 0, "y": 0, "width": width, "height": 80,
                "content": token, "font_family": "Solstice Sans", "font_size": 28, "line_height": 1.2,
            }
            self.assertGreater(_score_candidate([element], weights, 800, 200), 0.0, width)

    def test_height_uses_renderer_metrics_not_a_line_box_estimate(self):
        from engine.resize import _measured_lines, _score_candidate, _text_block_height

        content = "Hello"
        family = "Solstice Sans"
        size = 40.0
        measured = _measured_lines(content, family, "bold", None, size, 400)
        self.assertIsNotNone(measured)
        real = _text_block_height(measured[0], size, 1.2, family, "bold", None)
        estimate = measured[0] * size * 1.2
        self.assertGreater(real, estimate + 1.0)
        element = {
            "id": "hello", "type": "text", "x": 0, "y": 0, "width": 400,
            "height": (real + estimate) / 2,
            "content": content, "font_family": family, "font_weight": "bold",
            "font_size": size, "line_height": 1.2,
        }
        self.assertGreater(_score_candidate([element], {"hello": 1.0}, 800, 400), 0.0)

    def test_candidate_selection_keeps_the_grouped_button(self):
        from engine.resize import _score_candidate

        shape = {
            "id": "btn", "type": "shape", "group_id": "action",
            "x": 0, "y": 0, "width": 200, "height": 50,
        }
        inside = {
            "id": "label", "type": "text", "group_id": "action",
            "x": 20, "y": 10, "width": 150, "height": 30,
            "content": "Book tickets", "font_family": "Solstice Sans", "font_size": 16,
        }
        outside = dict(inside, x=700, y=10)
        weights = {"btn": 1.0, "label": 1.0}
        scored = {
            "grouped": _score_candidate([shape, inside], weights, 800, 400),
            "separated": _score_candidate([shape, outside], weights, 800, 400),
        }
        chosen = min(scored, key=scored.get)
        self.assertEqual(chosen, "grouped")
        self.assertLess(scored["grouped"], scored["separated"])


class CanvasCoverTests(unittest.TestCase):
    def test_cover_requires_fit_and_placement(self):
        from engine.resize import _covers_canvas

        contain = {"type": "image", "fit": "contain", "x": 0, "y": 0, "width": 1000, "height": 1000}
        placed = {"type": "image", "fit": "cover", "x": 0, "y": 0, "width": 960, "height": 960}
        shifted = {"type": "image", "fit": "cover", "x": -800, "y": 0, "width": 1000, "height": 1000}
        self.assertFalse(_covers_canvas(contain, 1000, 1000))
        self.assertTrue(_covers_canvas(placed, 1000, 1000))
        self.assertFalse(_covers_canvas(shifted, 1000, 1000))

    def test_official_backgrounds_cover_the_canvas_and_products_do_not(self):
        from engine.resize import _covers_canvas

        for output_name, input_name in INPUTS.items():
            task = json.loads((ROOT / "input" / input_name).read_text())
            canvas = task["source"]["canvas"]
            images = [element for element in task["source"]["elements"] if element["type"] == "image"]
            covers = [element["id"] for element in images if _covers_canvas(element, canvas["width"], canvas["height"])]
            self.assertEqual(covers, ["bg"], output_name)

    def test_the_scorer_ignores_only_a_real_background(self):
        from engine.resize import _score_candidate

        text = {
            "id": "line", "type": "text", "x": 40, "y": 40, "width": 200, "height": 40,
            "content": "Hello", "font_family": "Solstice Sans", "font_size": 24,
        }
        weights = {"photo": 1.0, "line": 1.0}
        contain = {
            "id": "photo", "type": "image", "fit": "contain",
            "x": 0, "y": 0, "width": 1000, "height": 1000,
        }
        cover = dict(contain, fit="cover", width=960, height=960)
        shifted = dict(contain, fit="cover", x=-800)
        self.assertEqual(_score_candidate([cover, text], weights, 1000, 1000), 0.0)
        self.assertGreater(_score_candidate([contain, text], weights, 1000, 1000), 0.0)
        self.assertGreater(_score_candidate([shifted, text], weights, 1000, 1000), 0.0)

    def test_a_full_contain_image_is_not_stretched_into_a_background(self):
        def photo(fit, x, y, width, height):
            return {
                "id": "photo", "type": "image", "asset_id": "solstice_bottle.png", "fit": fit,
                "x": x, "y": y, "width": width, "height": height, "z_index": 1,
            }

        target = {"width": 500, "height": 800}
        contain = resize(_design([photo("contain", 0, 0, 1000, 1000)], w=1000, h=1000), target)
        cover = resize(_design([photo("cover", 0, 0, 960, 960)], w=1000, h=1000), target)
        shifted = resize(_design([photo("cover", -800, 0, 1000, 1000)], w=1000, h=1000), target)
        filled = (0.0, 0.0, 500.0, 800.0)
        cover_box = next(element for element in cover["elements"] if element["id"] == "photo")
        self.assertEqual((cover_box["x"], cover_box["y"], cover_box["width"], cover_box["height"]), filled)
        for output in (contain, shifted):
            box = next(element for element in output["elements"] if element["id"] == "photo")
            self.assertNotEqual((box["x"], box["y"], box["width"], box["height"]), filled)


class EntrypointProvenanceTests(unittest.TestCase):
    def test_a_malformed_task_clears_the_previous_output(self):
        import io
        import tempfile
        import engine.main as main_mod

        previous = os.environ.get("CHAMELEON_TEXT_MODE")
        os.environ["CHAMELEON_TEXT_MODE"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                incoming, outgoing = Path(tmp) / "input", Path(tmp) / "output"
                incoming.mkdir()
                outgoing.mkdir()
                (outgoing / "broken_output.json").write_text('{"stale": true}\n')
                (incoming / "broken.json").write_text("{not json")
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch(
                    "sys.stderr", io.StringIO()
                ), patch("sys.stdout", io.StringIO()):
                    code = main_mod.main()
                self.assertNotEqual(code, 0)
                self.assertFalse((outgoing / "broken_output.json").exists())
        finally:
            if previous is None:
                os.environ.pop("CHAMELEON_TEXT_MODE", None)
            else:
                os.environ["CHAMELEON_TEXT_MODE"] = previous

    def test_model_judgment_and_layout_decision_are_separate(self):
        import io
        import tempfile
        import engine.main as main_mod

        source = _design([
            _text("title", "Weekend rack", 48, 80, w=700, h=80),
            _text("note", "Cotton jersey.", 18, 200, w=700, h=40),
        ])
        sidecar = {
            "status": "model",
            "model_judgment": False,
            "judgments": {
                "title": _judgment("Weekend rack", "headline", "primary", True),
                "note": _judgment("Cotton jersey.", "body", "low", False),
            },
            "text_ids": ["title", "note"],
            "attempts": [{"ok": True, "stats": {"wall_s": 1.0}}],
        }

        class FixedBackend:
            def identity(self):
                return {"name": "fixed"}

            def generate(self, system, user, response_schema):
                raise AssertionError("the test supplies the sidecar")

        previous = os.environ.get("CHAMELEON_TEXT_MODE")
        os.environ["CHAMELEON_TEXT_MODE"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                incoming, outgoing = Path(tmp) / "input", Path(tmp) / "output"
                incoming.mkdir()
                outgoing.mkdir()
                (incoming / "poster.json").write_text(json.dumps({
                    "source": source, "target_canvas": {"width": 1080, "height": 1080},
                }))
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "select_backend", return_value=(FixedBackend(), "packaged-gguf")
                ), patch.object(main_mod, "analyze", return_value=sidecar), patch(
                    "sys.stdout", io.StringIO()
                ):
                    code = main_mod.main()
                self.assertEqual(code, 0)
                saved = json.loads((outgoing / "poster_sidecar.json").read_text())
                self.assertTrue(saved["model_judgment"])
                self.assertIn(saved["layout_decision"], {"judged", "baseline_kept", "baseline"})
                self.assertFalse(saved["offer_moved"])
        finally:
            if previous is None:
                os.environ.pop("CHAMELEON_TEXT_MODE", None)
            else:
                os.environ["CHAMELEON_TEXT_MODE"] = previous

    def test_a_moved_offer_is_recorded_apart_from_the_layout_decision(self):
        import io
        import tempfile
        import engine.main as main_mod

        source = json.loads((ROOT / "eval" / "text_context" / "designs" / "d08.json").read_text())
        texts = {element["id"]: element["content"] for element in source["elements"] if element["type"] == "text"}
        judgments = {
            eid: _judgment(content, "body", "medium", False) for eid, content in texts.items()
        }
        judgments["t01"] = _judgment(texts["t01"], "discount", "high", True)
        judgments["t04"] = _judgment(texts["t04"], "price", "high", True)
        judgments["t02"] = _judgment(texts["t02"], "headline", "primary", True)
        sidecar = {
            "status": "model",
            "model_judgment": False,
            "judgments": judgments,
            "text_ids": sorted(judgments),
            "attempts": [{"ok": True, "stats": {"wall_s": 0.1}}],
        }

        class FixedBackend:
            def identity(self):
                return {"name": "fixed"}

            def generate(self, system, user, response_schema):
                raise AssertionError("the test supplies the sidecar")

        previous = os.environ.get("CHAMELEON_TEXT_MODE")
        os.environ["CHAMELEON_TEXT_MODE"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                incoming, outgoing = Path(tmp) / "input", Path(tmp) / "output"
                incoming.mkdir()
                outgoing.mkdir()
                (incoming / "sale.json").write_text(json.dumps({
                    "source": source, "target_canvas": {"width": 1200, "height": 628},
                }))
                with patch.object(main_mod, "INPUT_DIR", incoming), patch.object(main_mod, "OUTPUT_DIR", outgoing), patch.object(
                    main_mod, "select_backend", return_value=(FixedBackend(), "packaged-gguf")
                ), patch.object(main_mod, "analyze", return_value=sidecar), patch(
                    "sys.stdout", io.StringIO()
                ):
                    code = main_mod.main()
                self.assertEqual(code, 0)
                saved = json.loads((outgoing / "sale_sidecar.json").read_text())
                output = json.loads((outgoing / "sale_output.json").read_text())
                offer = next(element for element in output["elements"] if element["id"] == "t01")
                self.assertEqual(saved["layout_decision"], "baseline_kept")
                self.assertTrue(saved["offer_moved"])
                self.assertGreater(offer["font_size"], 40)
                self.assertIn("offer moved", main_mod._status_label(saved))
                self.assertNotIn("judged", saved["layout_decision"])
        finally:
            if previous is None:
                os.environ.pop("CHAMELEON_TEXT_MODE", None)
            else:
                os.environ["CHAMELEON_TEXT_MODE"] = previous


class SameSizeGateTests(unittest.TestCase):
    def _design(self, title_size, action_size):
        return {
            "canvas": {"width": 400, "height": 400},
            "elements": [
                {"id": "title", "type": "text", "content": "Sunday market", "font_size": title_size,
                 "x": 20, "y": 20, "width": 300, "height": 80},
                {"id": "action", "type": "text", "content": "Get directions", "font_size": action_size,
                 "x": 20, "y": 200, "width": 220, "height": 40},
            ],
        }

    def test_a_same_size_canvas_keeps_a_clearer_essential_line(self):
        from engine.resize import _keep_judged_layout

        source = self._design(40, 16)
        baseline = self._design(40, 16)
        judged = self._design(39, 24)
        guidance = type("Guidance", (), {"essential_ids": {"title", "action"}, "protected_ids": set()})()
        self.assertTrue(_keep_judged_layout(source, source["canvas"], baseline, judged, guidance, {}))

    def test_a_same_size_canvas_keeps_the_baseline_when_nothing_grows(self):
        from engine.resize import _keep_judged_layout

        source = self._design(40, 16)
        baseline = self._design(40, 16)
        judged = self._design(40, 17)
        guidance = type("Guidance", (), {"essential_ids": {"title", "action"}, "protected_ids": set()})()
        self.assertFalse(_keep_judged_layout(source, source["canvas"], baseline, judged, guidance, {}))

    def test_a_same_size_canvas_rejects_a_real_essential_shrink(self):
        from engine.resize import _keep_judged_layout

        source = self._design(40, 16)
        baseline = self._design(40, 16)
        judged = self._design(30, 28)
        guidance = type("Guidance", (), {"essential_ids": {"title", "action"}, "protected_ids": set()})()
        self.assertFalse(_keep_judged_layout(source, source["canvas"], baseline, judged, guidance, {}))


class SecondaryYieldTests(unittest.TestCase):
    def _pair(self, base_price, base_name, judged_price, judged_name, width, height):
        def design(price, name):
            return {
                "canvas": {"width": width, "height": height},
                "elements": [
                    {"id": "price", "type": "text", "content": "$12", "font_size": price,
                     "font_family": "Solstice Sans", "x": 20, "y": 20, "width": 900, "height": 200},
                    {"id": "name", "type": "text", "content": "Summer pears", "font_size": name,
                     "font_family": "Solstice Serif", "x": 20, "y": 400, "width": 1000, "height": 220},
                ],
            }
        source = design(48, 36)
        source["canvas"] = {"width": 1080, "height": 1080}
        return source, design(base_price, base_name), design(judged_price, judged_name)

    def test_a_high_price_does_not_yield_half_its_size_to_a_headline(self):
        from engine.resize import _keep_judged_layout

        source, baseline, judged = self._pair(76, 44, 50, 90, 1200, 628)
        guidance = type("Guidance", (), {
            "essential_ids": {"price", "name"},
            "protected_ids": {"price"},
            "headline": {"id": "name"},
            "slots": {},
        })()
        judgments = {
            "price": _judgment("$12", "price", "high", True),
            "name": _judgment("Summer pears", "headline", "primary", True),
        }
        self.assertFalse(_keep_judged_layout(
            source, {"width": 1200, "height": 628}, baseline, judged, guidance, judgments,
        ))

    def test_a_low_price_can_still_yield_to_a_headline(self):
        from engine.resize import _keep_judged_layout

        source, baseline, judged = self._pair(76, 44, 50, 90, 1200, 628)
        guidance = type("Guidance", (), {
            "essential_ids": {"price", "name"},
            "protected_ids": set(),
            "headline": {"id": "name"},
            "slots": {},
        })()
        judgments = {
            "price": _judgment("$12", "price", "low", True),
            "name": _judgment("Summer pears", "headline", "primary", True),
        }
        self.assertTrue(_keep_judged_layout(
            source, {"width": 1200, "height": 628}, baseline, judged, guidance, judgments,
        ))

    def test_a_true_resize_keeps_a_headline_that_overtakes_a_price(self):
        from engine.resize import _keep_judged_layout

        source, baseline, judged = self._pair(76, 44, 50, 90, 1200, 628)
        guidance = type("Guidance", (), {
            "essential_ids": {"price", "name"},
            "protected_ids": {"price"},
            "headline": {"id": "name"},
        })()
        self.assertTrue(_keep_judged_layout(
            source, {"width": 1200, "height": 628}, baseline, judged, guidance, {},
        ))

    def test_a_headline_still_cannot_shrink_to_lift_a_secondary_line(self):
        from engine.resize import _keep_judged_layout

        source, baseline, judged = self._pair(40, 80, 70, 40, 1200, 628)
        guidance = type("Guidance", (), {
            "essential_ids": {"price", "name"},
            "protected_ids": set(),
            "headline": {"id": "name"},
        })()
        self.assertFalse(_keep_judged_layout(
            source, {"width": 1200, "height": 628}, baseline, judged, guidance, {},
        ))

    def test_a_secondary_line_cannot_fall_below_half_its_baseline(self):
        from engine.resize import _keep_judged_layout

        source, baseline, judged = self._pair(46, 40, 18, 64, 1080, 1350)
        guidance = type("Guidance", (), {
            "essential_ids": {"price", "name"},
            "protected_ids": set(),
            "headline": {"id": "name"},
        })()
        self.assertFalse(_keep_judged_layout(
            source, {"width": 1080, "height": 1350}, baseline, judged, guidance, {},
        ))


class OfferHierarchyTests(unittest.TestCase):
    def _sale_card(self):
        return _design([
            _text("price", "18", 48, 140, w=220, h=70),
            _text("name", "Harbor coat", 36, 280, w=640, h=60),
            _text("note", "Ends Sunday", 16, 420, w=240, h=32),
        ])

    def _sizes(self, design, target, judgments):
        out = resize(design, target, judgments=judgments)
        return {element["id"]: element["font_size"] for element in out["elements"] if element["type"] == "text"}

    def test_a_louder_price_leads_the_product_name(self):
        judgments = {
            "price": _judgment("18", "price", "primary", True),
            "name": _judgment("Harbor coat", "product_name", "medium", True),
            "note": _judgment("Ends Sunday", "date_time", "low", True),
        }
        sizes = self._sizes(self._sale_card(), {"width": 1200, "height": 628}, judgments)
        self.assertGreater(sizes["price"], sizes["name"])
        self.assertEqual(layout_bridge.last_guidance.slots.get("name"), "body")

    def test_a_high_price_keeps_the_lead_on_a_same_size_canvas(self):
        judgments = {
            "price": _judgment("18", "price", "high", True),
            "name": _judgment("Harbor coat", "headline", "primary", True),
            "note": _judgment("Ends Sunday", "date_time", "low", True),
        }
        design = self._sale_card()
        sizes = self._sizes(design, {"width": 1080, "height": 1080}, judgments)
        self.assertGreaterEqual(sizes["price"], sizes["name"])

    def test_a_product_led_card_keeps_the_name_ahead_of_a_smaller_price(self):
        design = _design([
            _text("name", "Harbor coat", 64, 120, w=700, h=80),
            _text("price", "18", 22, 280, w=180, h=40),
            _text("note", "Wool lined.", 16, 400, w=400, h=32),
        ])
        judgments = {
            "name": _judgment("Harbor coat", "headline", "primary", True),
            "price": _judgment("18", "price", "low", True),
            "note": _judgment("Wool lined.", "body", "low", False),
        }
        sizes = self._sizes(design, {"width": 1200, "height": 628}, judgments)
        self.assertGreater(sizes["name"], sizes["price"])

    def test_a_primary_product_name_still_leads_a_quieter_price(self):
        judgments = {
            "price": _judgment("18", "price", "medium", True),
            "name": _judgment("Harbor coat", "product_name", "primary", True),
            "note": _judgment("Ends Sunday", "date_time", "low", True),
        }
        sizes = self._sizes(self._sale_card(), {"width": 1200, "height": 628}, judgments)
        self.assertGreater(sizes["name"], sizes["price"])
        self.assertEqual(layout_bridge.last_guidance.slots.get("name"), "headline")


class BusyBackgroundCenterTests(unittest.TestCase):
    """A calmer patch must not slide a center the layout already chose."""

    def _gray(self, width, height, paint):
        from PIL import Image

        image = Image.new("L", (width, height), 180)
        paint(image)
        return image

    def _noise(self, image, box, seed):
        import random

        rng = random.Random(seed)
        x0, y0, w, h = (int(v) for v in box)
        for py in range(max(0, y0), min(image.height, y0 + h)):
            for px in range(max(0, x0), min(image.width, x0 + w)):
                image.putpixel((px, py), rng.randrange(0, 256))

    def _text(self, x, y, width, height, align):
        return {
            "id": "line", "type": "text", "x": x, "y": y, "width": width, "height": height,
            "content": "Centered line", "font_family": "Solstice Sans", "font_size": 24,
            "font_weight": "bold", "text_align": align, "line_height": 1.2,
        }

    def test_a_canvas_centered_line_does_not_slide_sideways(self):
        from engine.resize import _avoid_busy_background

        tw, th = 400.0, 220.0
        element = self._text(100.0, 90.0, 200.0, 50.0, "center")

        def paint(image):
            # Noise sits in the lower part of the centered box. The same row
            # to the left is calm, which is the slide the lock has to refuse.
            # The row above clears the noise, so a vertical nudge can still win.
            self._noise(image, (210, 125, 80, 16), 1)

        moved = _avoid_busy_background(
            [element], {"line"}, {}, self._gray(int(tw), int(th), paint), tw, th,
        )[0]
        center = moved["x"] + moved["width"] / 2.0
        self.assertLess(abs(center - tw / 2.0), 8.0)
        self.assertEqual(moved["x"], 100.0)
        self.assertEqual(moved["width"], 200.0)
        self.assertNotEqual(moved["y"], 90.0)

    def test_a_center_aligned_column_can_still_move_sideways(self):
        from engine.resize import _avoid_busy_background

        tw, th = 400.0, 160.0
        element = self._text(20.0, 40.0, 90.0, 36.0, "center")

        def paint(image):
            self._noise(image, (20, 40, 90, 36), 3)

        moved = _avoid_busy_background(
            [element], {"line"}, {}, self._gray(int(tw), int(th), paint), tw, th,
        )[0]
        self.assertNotEqual(moved["x"], 20.0)
        self.assertGreater(abs((moved["x"] + moved["width"] / 2.0) - tw / 2.0), 8.0)

    def test_left_aligned_text_on_the_canvas_center_can_still_move(self):
        from engine.resize import _avoid_busy_background

        tw, th = 400.0, 160.0
        element = self._text(100.0, 50.0, 200.0, 40.0, "left")

        def paint(image):
            self._noise(image, (180, 50, 120, 40), 4)

        moved = _avoid_busy_background(
            [element], {"line"}, {}, self._gray(int(tw), int(th), paint), tw, th,
        )[0]
        self.assertNotEqual(moved["x"], 100.0)

    def test_a_collision_blocks_the_vertical_nudge(self):
        from engine.resize import _avoid_busy_background

        tw, th = 400.0, 220.0
        element = self._text(100.0, 90.0, 200.0, 50.0, "center")

        def paint(image):
            self._noise(image, (210, 125, 80, 16), 5)

        blocker = {"other": (0.0, 70.0, 400.0, 50.0)}
        moved = _avoid_busy_background(
            [element], {"line"}, blocker, self._gray(int(tw), int(th), paint), tw, th,
        )[0]
        self.assertEqual(moved["x"], 100.0)
        self.assertEqual(moved["y"], 90.0)

    def test_official_level1_headline_stays_on_the_source_center(self):
        task = json.loads((ROOT / "input" / "chameleon_sample_level1_warmup.json").read_text())
        judgments = {
            "headline": _judgment("Chase the Light", "headline", "primary", True),
            "cta": _judgment("Shop the Summer Line", "call_to_action", "high", True),
        }
        out = resize(task["source"], task["target_canvas"], judgments=judgments)
        headline = next(element for element in out["elements"] if element["id"] == "headline")
        action = next(element for element in out["elements"] if element["id"] == "cta")
        canvas_center = out["canvas"]["width"] / 2.0
        self.assertEqual(headline["content"], "Chase the Light")
        self.assertEqual(headline["text_align"], "center")
        self.assertLess(abs(headline["x"] + headline["width"] / 2.0 - canvas_center), 8.0)
        self.assertLess(abs(action["x"] + action["width"] / 2.0 - canvas_center), 8.0)
        self.assertEqual(len(out["elements"]), len(task["source"]["elements"]))


if __name__ == "__main__":
    unittest.main()
