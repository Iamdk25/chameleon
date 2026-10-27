import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.layout_eval import checks

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "chameleon_design_schema_v0.9.json").read_text())
DRIVER = ROOT / "tools" / "layout_eval" / "driver.py"

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

try:
    import skia  # noqa: F401
    import cr_renderer  # noqa: F401
    HAVE_RENDERER = True
except Exception:
    HAVE_RENDERER = False


def text(eid, x, y, w, h, size=20, **kw):
    el = {"id": eid, "type": "text", "x": x, "y": y, "width": w, "height": h, "content": eid,
          "font_family": "Solstice Sans", "font_size": size, "color": "rgba(0,0,0,1)"}
    el.update(kw)
    return el


def shape(eid, x, y, w, h, **kw):
    el = {"id": eid, "type": "shape", "x": x, "y": y, "width": w, "height": h,
          "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)"}
    el.update(kw)
    return el


def design(w, h, elements):
    return {"canvas": {"width": w, "height": h}, "elements": elements}


def metric(w, h, size=20):
    return {"font_size": size, "drawn_px": int(size), "lines": 1, "max_line_width": w,
            "block_height": h, "local_extent": [0.0, 0.0, w, h]}


class GeometryTests(unittest.TestCase):
    def test_rotated_aabb_quarter_turn(self):
        x, y, w, h = checks.rotated_aabb(0, 0, 100, 20, 90)
        self.assertAlmostEqual(w, 20)
        self.assertAlmostEqual(h, 100)
        self.assertAlmostEqual(x, 40)
        self.assertAlmostEqual(y, -40)

    def test_kendall_tau(self):
        self.assertEqual(checks.kendall_tau_b([3, 2, 1], [30, 20, 10]), 1.0)
        self.assertEqual(checks.kendall_tau_b([3, 2, 1], [10, 20, 30]), -1.0)
        self.assertIsNone(checks.kendall_tau_b([1], [1]))


@unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema not installed; contract checks would report 'schema not checked'")
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.source = design(100, 100, [text("a", 10, 10, 50, 20), shape("b", 0, 0, 10, 10)])
        self.target = {"width": 200, "height": 50}

    def out(self, elements):
        return design(200, 50, elements)

    def test_clean_output_passes(self):
        out = self.out(copy.deepcopy(self.source["elements"]))
        self.assertEqual(checks.check_contract(self.source, self.target, out, SCHEMA), [])

    def test_missing_id_and_changed_content_fail(self):
        a = copy.deepcopy(self.source["elements"][0])
        a["content"] = "rewritten"
        failures = checks.check_contract(self.source, self.target, self.out([a]), SCHEMA)
        self.assertTrue(any(f.startswith("ids: element count") for f in failures))
        self.assertTrue(any("protected: a.content" in f for f in failures))

    def test_schema_violation_fails(self):
        els = copy.deepcopy(self.source["elements"])
        els[0]["opacity"] = 2
        failures = checks.check_contract(self.source, self.target, self.out(els), SCHEMA)
        self.assertTrue(any(f.startswith("schema:") for f in failures))

    def test_wrong_canvas_fails(self):
        out = design(100, 100, copy.deepcopy(self.source["elements"]))
        failures = checks.check_contract(self.source, self.target, out, SCHEMA)
        self.assertTrue(any(f.startswith("canvas:") for f in failures))

    def test_changed_field_classification(self):
        els = copy.deepcopy(self.source["elements"])
        els[0].update(x=99, font_size=12, opacity=0.5, color="rgba(1,1,1,1)")
        els[1].update(z_index=4, rotation=3)
        changes = checks.changed_fields(self.source, self.out(els))
        self.assertEqual(changes["a"]["x"]["kind"], "faq_q1_allowed")
        self.assertEqual(changes["a"]["font_size"]["kind"], "faq_q1_allowed")
        self.assertEqual(changes["a"]["opacity"]["kind"], "rule_question")
        self.assertEqual(changes["a"]["color"]["kind"], "other")
        self.assertEqual(changes["b"]["z_index"]["kind"], "rule_question")
        self.assertEqual(changes["b"]["rotation"]["kind"], "rule_question")
        self.assertEqual(len(checks.rule_questions(changes)), 4)

    def test_default_valued_fields_are_not_changes(self):
        els = copy.deepcopy(self.source["elements"])
        els[0]["opacity"] = 1
        els[0]["rotation"] = 0
        self.assertEqual(checks.changed_fields(self.source, self.out(els)), {})


class BoundsAndOverlapTests(unittest.TestCase):
    def test_source_bleed_is_carried_not_new(self):
        src = design(100, 100, [shape("bleed", -10, -10, 50, 50), shape("inside", 10, 10, 10, 10)])
        out = design(200, 50, [shape("bleed", -5, -5, 20, 20), shape("inside", 190, 10, 20, 10)])
        b = checks.bounds_report(src, out, {}, {})
        self.assertEqual([i["id"] for i in b["rotation_aware_off_canvas_carried_source_bleed"]], ["bleed"])
        self.assertEqual([i["id"] for i in b["rotation_aware_off_canvas_new"]], ["inside"])
        self.assertEqual(sorted(b["starter_self_check_off_canvas"]), ["bleed", "inside"])

    def test_rotation_can_push_a_box_off_canvas(self):
        src = design(100, 100, [shape("r", 10, 10, 20, 20)])
        out = design(100, 100, [shape("r", 0, 0, 100, 20, rotation=30)])
        b = checks.bounds_report(src, out, {}, {})
        self.assertEqual(b["starter_self_check_off_canvas"], [])
        self.assertEqual([i["id"] for i in b["rotation_aware_off_canvas_new"]], ["r"])

    def test_text_ink_clipped_by_canvas(self):
        src = design(100, 100, [text("t", 10, 10, 50, 20)])
        out = design(100, 30, [text("t", 10, 10, 50, 20)])
        b = checks.bounds_report(src, out, {"t": metric(40, 18)}, {"t": metric(40, 40)})
        self.assertEqual([i["id"] for i in b["text_clipped_by_canvas_new"]], ["t"])

    def test_new_overlap_detected_and_groups_and_background_ignored(self):
        bg = {"id": "bg", "type": "image", "x": 0, "y": 0, "width": 100, "height": 100,
              "asset_id": "x.png", "fit": "cover"}
        src = design(100, 100, [bg, text("t", 0, 0, 40, 20), shape("s", 60, 60, 20, 20),
                                text("g1", 0, 50, 20, 10, group_id="g"), shape("g2", 0, 50, 20, 10, group_id="g")])
        out = design(100, 100, [bg, text("t", 0, 0, 40, 20), shape("s", 10, 0, 20, 20),
                                text("g1", 0, 50, 20, 10, group_id="g"), shape("g2", 0, 50, 20, 10, group_id="g")])
        m = {"t": metric(40, 20), "g1": metric(20, 10)}
        ov = checks.overlap_report(src, out, m, m, {})
        self.assertEqual([o["pair"] for o in ov["new"]], [["s", "t"]])
        self.assertEqual(len(ov["new_text_involved"]), 1)

    def test_preexisting_overlap_is_carried(self):
        src = design(100, 100, [text("t", 0, 0, 40, 20), shape("s", 10, 0, 20, 20)])
        m = {"t": metric(40, 20)}
        ov = checks.overlap_report(src, copy.deepcopy(src), m, m, {})
        self.assertEqual(ov["new"], [])
        self.assertEqual(len(ov["carried_from_source"]), 1)

    def test_group_label_spilling_outside_its_shape(self):
        out = design(100, 100, [shape("pill", 0, 0, 30, 10, group_id="g"), text("lbl", 0, 0, 30, 10, group_id="g")])
        problems = checks.group_report(out, {"lbl": metric(45, 10)})
        self.assertEqual(problems[0]["text"], "lbl")
        self.assertGreater(problems[0]["spill_px"]["right"], 10)

    def test_text_box_overflow_new_vs_source(self):
        src = design(100, 100, [text("t", 0, 0, 50, 20)])
        out = design(100, 100, [text("t", 0, 0, 50, 20)])
        fit = checks.text_fit_report(src, out, {"t": metric(40, 18)}, {"t": metric(60, 18)})
        self.assertTrue(fit["t"]["overflows_box"])
        self.assertFalse(fit["t"]["source_overflowed_box"])


class ImageAndHierarchyTests(unittest.TestCase):
    def test_fill_distortion_and_cover_crop(self):
        img = {"id": "i", "type": "image", "x": 0, "y": 0, "width": 100, "height": 100, "asset_id": "a.png"}
        src = design(100, 100, [dict(img, fit="fill"), dict(img, id="c", fit="cover")])
        out = design(200, 100, [dict(img, fit="fill", width=200), dict(img, id="c", fit="cover", width=200)])
        rows = {r["id"]: r for r in checks.image_report(src, out, {"a.png": {"width": 10, "height": 10}})}
        self.assertAlmostEqual(rows["i"]["pixel_distortion"], 1.0)
        self.assertEqual(rows["c"]["pixel_distortion"], 0.0)
        self.assertEqual(rows["c"]["cover_visible_fraction_output"], 0.5)

    def test_size_inversion(self):
        src = design(100, 100, [text("big", 0, 0, 10, 10, size=40), text("small", 0, 0, 10, 10, size=20)])
        out = design(100, 100, [text("big", 0, 0, 10, 10, size=10), text("small", 0, 0, 10, 10, size=15)])
        h = checks.hierarchy_report(src, out)
        self.assertEqual(h["kendall_tau_b_font_size"], -1.0)
        self.assertEqual(h["size_inversions"][0]["larger_in_source"], "big")


class ProvenanceTests(unittest.TestCase):
    def result(self, root, **kw):
        kit = f"{root}/brand_kit"
        r = {"engine_module": f"{root}/engine/resize.py", "renderer_module": f"{root}/renderer/converter/convert.py",
             "engine_fonts_dir": f"{kit}/fonts", "engine_assets_dir": None, "font_manager_loaded": True,
             "out_dir": "/tmp/out", "asset_native": {},
             "opens": [{"phase": "resize", "path": f"{kit}/fonts/A.ttf"},
                       {"phase": "render_output", "path": f"{kit}/fonts/A.ttf"}]}
        r.update(kw)
        return r

    def test_clean_provenance(self):
        src = design(10, 10, [text("t", 0, 0, 5, 5)])
        self.assertEqual(checks.provenance_failures(self.result("/ck"), Path("/ck"), src), [])

    def test_font_fallback_and_foreign_paths_fail(self):
        src = design(10, 10, [text("t", 0, 0, 5, 5)])
        r = self.result("/ck", font_manager_loaded=False, engine_fonts_dir="/tmp/brand_kit/fonts",
                        opens=[{"phase": "render_output", "path": "/other/brand_kit/fonts/A.ttf"}])
        failures = checks.provenance_failures(r, Path("/ck"), src)
        joined = "\n".join(failures)
        self.assertIn("font manager did not load", joined)
        self.assertIn("engine_fonts_dir", joined)
        self.assertIn("outside /ck/brand_kit", joined)
        self.assertIn("no brand font file was opened during resize", joined)

    def test_background_asset_must_be_read_by_asset_aware_engine(self):
        bg = {"id": "bg", "type": "image", "x": 0, "y": 0, "width": 10, "height": 10,
              "asset_id": "bg.png", "fit": "cover"}
        r = self.result("/ck", engine_assets_dir="/ck/brand_kit/assets",
                        asset_native={"bg.png": {"width": 1, "height": 1}})
        r["opens"].append({"phase": "render_output", "path": "/ck/brand_kit/assets/bg.png"})
        failures = checks.provenance_failures(r, Path("/ck"), design(10, 10, [bg]))
        self.assertTrue(any("never opened background asset" in f for f in failures))


@unittest.skipUnless(HAVE_RENDERER, "real renderer (skia, cr_renderer) not installed in this interpreter")
class DriverIntegrationTests(unittest.TestCase):
    """Runs the real engine in this checkout. The negative control copies
    engine/ and renderer/ without brand_kit/, which is what loading
    resize.py outside a checkout amounts to: the engine silently falls
    back, and provenance must catch it."""

    def run_driver(self, checkout: Path, case: Path, out: Path) -> dict:
        subprocess.run([sys.executable, str(DRIVER), "--checkout", str(checkout), "--case", str(case),
                        "--out", str(out)], capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
        return json.loads((out / "result.json").read_text())

    def test_real_checkout_passes_and_broken_paths_fail(self):
        case = ROOT / "input" / "chameleon_sample_level1_warmup.json"
        task = json.loads(case.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good = self.run_driver(ROOT, case, tmp / "good")
            self.assertEqual(good["status"], "ok")
            self.assertEqual(checks.provenance_failures(good, ROOT, task["source"]), [])
            output = json.loads((tmp / "good" / "output.json").read_text())
            report = checks.run_all(task, output, good, SCHEMA, ROOT)
            self.assertEqual(report["contract"], [])

            broken = tmp / "broken"
            shutil.copytree(ROOT / "engine", broken / "engine")
            shutil.copytree(ROOT / "renderer", broken / "renderer")
            bad = self.run_driver(broken, case, tmp / "bad")
            self.assertFalse(bad["font_manager_loaded"])
            failures = checks.provenance_failures(bad, broken, task["source"])
            self.assertTrue(any("font manager did not load" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
