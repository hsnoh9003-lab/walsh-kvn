"""Offline checks for shared local/Zenodo data resolution and explicit CFD inputs."""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Source import data_locations as locations
from Source.CFD.inputs import STEADY_INPUT, steady_input
from Source.CFD.fem import run_fem
from Source.ROM import model


def dataset(root, relative, contents=b'input'):
    target = root/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)
    with (root/'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['original_path', 'published_path'])
        writer.writeheader()
        writer.writerow({'original_path': STEADY_INPUT, 'published_path': relative})
    return target


class DataPathTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        os.environ.pop('KVN_SOURCE_DATA', None)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        locations.source_data_root.cache_clear()
        locations._original_to_published.cache_clear()

    def tearDown(self):
        self.environment.stop()
        locations.source_data_root.cache_clear()
        locations._original_to_published.cache_clear()
        self.temporary.cleanup()

    def test_local_cache_and_download_use_the_same_manifest(self):
        code = self.root/'Codes'
        local = self.root/'Source_Data'
        cached = self.root/'cache/Source_Data'
        expected_local = dataset(local, 'relocated/local.npz')
        expected_cache = dataset(cached, 'relocated/cache.npz')
        with (
            patch.object(locations, 'REPO_ROOT', code),
            patch.object(locations, 'cached_path', return_value=cached),
            patch.object(locations, 'download_source_data', return_value=cached) as download,
        ):
            self.assertEqual(steady_input(), expected_local)
            download.assert_not_called()
            (local/'manifest.csv').unlink()
            locations.source_data_root.cache_clear()
            locations._original_to_published.cache_clear()
            self.assertEqual(steady_input(), expected_cache)
            download.assert_not_called()
            # The downloader itself is unchanged; this only tests resolver dispatch.
            with patch.object(locations, 'cached_path', return_value=self.root/'empty_cache'):
                locations.source_data_root.cache_clear()
                locations._original_to_published.cache_clear()
                self.assertEqual(steady_input(), expected_cache)
                download.assert_called_once_with()

    def test_explicit_override_invalidates_both_caches(self):
        first, second = self.root/'first', self.root/'second'
        a = dataset(first, 'one.npz')
        b = dataset(second, 'different/two.npz')
        locations.set_source_data_root(first)
        self.assertEqual(steady_input(), a)
        locations.set_source_data_root(second)
        self.assertEqual(steady_input(), b)
        with patch.object(locations, 'download_source_data') as download:
            with self.assertRaises(FileNotFoundError):
                locations.set_source_data_root(self.root/'missing')
            download.assert_not_called()
        self.assertEqual(steady_input(), b)

    def test_launcher_uses_manifest_after_an_earlier_lookup(self):
        a = self.root/'a'; b = self.root/'b'
        dataset(a, 'old.npz')
        selected = dataset(b, 'moved/steady.npz')
        locations.set_source_data_root(a)
        steady_input()
        with (
            patch('sys.argv', ['run_fem', '--reference-replay', '--data-root', str(b), '--end', '.04']),
            patch.object(run_fem.subprocess, 'run') as run,
        ):
            run_fem.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--input')+1], str(selected))

    def test_generated_inputs_bypass_published_dataset(self):
        explicit = self.root/'generated_Re200.npz'
        explicit.write_bytes(b'generated')
        with patch('Source.CFD.inputs.source_file') as source:
            self.assertEqual(steady_input('fine', 200., explicit), explicit)
            source.assert_not_called()
            with self.assertRaisesRegex(FileNotFoundError, 'only Re=100'):
                steady_input('coarse', 200.)
            source.assert_not_called()

    def test_model_default_tracks_the_selected_dataset(self):
        original = locations.shared_data('CFD_ROM', 'Model')
        with np.load(original/'reported_model_scaling.npz') as data:
            scaling = {key: data[key] for key in data.files}
        for index in (1, 2):
            data_root = self.root/f'data{index}'
            dataset(data_root, 'unused.npz')
            folder = data_root/'Shared_source_data/CFD_ROM/Model'
            folder.mkdir(parents=True)
            modified = dict(scaling)
            modified['coefficients'] = scaling['coefficients']*index
            np.savez(folder/'reported_model_scaling.npz', **modified)
            locations.set_source_data_root(data_root)
            np.testing.assert_array_equal(model.load('deployed').physical_coefficients, modified['coefficients'])

    def test_imports_and_help_do_not_need_a_dataset(self):
        env = dict(os.environ, KVN_SOURCE_DATA=str(self.root/'missing'))
        command = '''
from Source.POD_SINDy.fit import fit
from Source.ROM.model import load
from Source.CFD import physical_audit
from Source.CFD.fem import simulate
'''
        subprocess.run([sys.executable, '-B', '-c', command], env=env, check=True)
        subprocess.run(
            [sys.executable, '-B', '-m', 'Source.CFD.physical_audit', '--help'],
            env=env, check=True, stdout=subprocess.DEVNULL,
        )


if __name__ == '__main__':
    unittest.main()
