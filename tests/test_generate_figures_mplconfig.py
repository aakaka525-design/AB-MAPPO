import os
import tempfile
import unittest
from unittest.mock import patch

import generate_figures


class TestGenerateFiguresMPLConfig(unittest.TestCase):
    def test_sets_mplconfigdir_when_unset(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MPLCONFIGDIR", None)
                path = generate_figures._ensure_writable_mplconfigdir(candidates=[td])
                self.assertEqual(path, td)
                self.assertEqual(os.environ.get("MPLCONFIGDIR"), td)
                probe = os.path.join(td, ".write_probe")
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("ok")

    def test_keeps_existing_mplconfigdir(self):
        with tempfile.TemporaryDirectory() as existing, tempfile.TemporaryDirectory() as other:
            with patch.dict(os.environ, {"MPLCONFIGDIR": existing}, clear=False):
                path = generate_figures._ensure_writable_mplconfigdir(candidates=[other])
                self.assertEqual(path, existing)
                self.assertEqual(os.environ.get("MPLCONFIGDIR"), existing)


if __name__ == "__main__":
    unittest.main()
