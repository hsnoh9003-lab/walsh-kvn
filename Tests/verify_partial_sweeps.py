"""Check that rerunning one resource condition preserves the others."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Reproduction.sweep_output import staged_sweep


class PartialSweepTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.table = self.output / "designs.csv"
        self.initial = (
            "scheme,p_phys,result\n"
            "A,0.0003,0.10000000000000001\n"
            "A,0.0005,old\n"
            "B,0.0005,other circuit\n"
        )
        self.table.write_text(self.initial, encoding="utf-8")

    def test_retry_preserves_other_rates_and_circuits(self):
        for _ in range(2):
            with staged_sweep(self.output, [0.0005], ["A"]) as stage:
                (stage / self.table.name).write_text(
                    "scheme,p_phys,result\nA,0.0005,new\n", encoding="utf-8"
                )
            with self.table.open(encoding="utf-8", newline="") as stream:
                results = {(r["scheme"], r["p_phys"]): r["result"] for r in csv.DictReader(stream)}
            self.assertEqual(results, {
                ("A", "0.0003"): "0.10000000000000001",
                ("A", "0.0005"): "new",
                ("B", "0.0005"): "other circuit",
            })

    def test_unrequested_rows_abort_the_publication(self):
        other = self.output / "another.csv"
        with self.assertRaises(ValueError):
            with staged_sweep(self.output, [0.0005], ["A"]) as stage:
                (stage / other.name).write_text(
                    "scheme,p_phys,result\nA,0.0005,valid\n", encoding="utf-8"
                )
                (stage / self.table.name).write_text(
                    "scheme,p_phys,result\nA,0.0003,unexpected\n", encoding="utf-8"
                )
        self.assertEqual(self.table.read_text(encoding="utf-8"), self.initial)
        self.assertFalse(other.exists())


class PublishedFactoryComparisonTest(unittest.TestCase):
    def test_figure_uses_scheme_a_with_both_circuits_present(self):
        from Reproduction.update_litinski_s8b import LABELS, published_wake_designs

        rows = [
            {"paper": paper, "p_phys": str(rate / 10000), "scheme": scheme}
            for scheme in ("A", "B") for paper in LABELS for rate in range(2, 10)
        ]
        selected = published_wake_designs(rows)
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(row["scheme"] == "A" for row in selected.values()))
        self.assertEqual(selected, published_wake_designs(reversed(rows)))
        with self.assertRaises(ValueError):
            published_wake_designs(rows + [rows[0]])

    def test_importing_the_factory_search_preserves_other_core_widths(self):
        from Source.RE import model

        count = model.core_tile_count
        from Reproduction import generate_litinski_data

        self.assertIs(model.core_tile_count, count)
        for width, patches in [(9, 28), (25, 66)]:
            config = model.ModelConfig(n_L=width, data_block="fast")
            self.assertEqual(model.core_tile_count(19, config)["data_block_tiles"], patches)


if __name__ == "__main__":
    unittest.main()
