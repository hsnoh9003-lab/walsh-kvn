"""Check that CFD re-extraction cannot overwrite the published Source_Data."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Source.CFD import physical_audit


class AuditInputTests(unittest.TestCase):
    def test_fresh_inputs_follow_output_for_both_extraction_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            for options in (['--from-cfd', '/unused/raw_cfd'], ['--full-cfd']):
                with self.subTest(options=options):
                    with (
                        patch('sys.argv', ['physical_audit', *options, '--output', str(output)]),
                        patch.object(physical_audit, 'extract') as extract,
                        patch.object(physical_audit, 'calculate') as calculate,
                        patch.object(physical_audit, 'run_cfd'),
                    ):
                        physical_audit.main()
                        self.assertEqual(extract.call_args.args[1], output/'inputs')
                        self.assertEqual(calculate.call_args.args[0], output/'inputs')

    def test_published_inputs_are_read_by_default(self):
        with (
            patch('sys.argv', ['physical_audit']),
            patch.object(physical_audit, 'extract') as extract,
            patch.object(physical_audit, 'calculate') as calculate,
        ):
            physical_audit.main()
            extract.assert_not_called()
            self.assertEqual(calculate.call_args.args[0], physical_audit.DATA)

    def test_overwrite_is_rejected_before_starting_cfd(self):
        with (
            patch('sys.argv', ['physical_audit', '--full-cfd', '--inputs', str(physical_audit.DATA)]),
            patch.object(physical_audit, 'extract') as extract,
            patch.object(physical_audit, 'run_cfd') as run_cfd,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                physical_audit.main()
            self.assertEqual(error.exception.code, 2)
            extract.assert_not_called()
            run_cfd.assert_not_called()


if __name__ == '__main__':
    unittest.main()
