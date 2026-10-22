import io
import runpy
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from src.model import DiagnosticModel

class SignedComparisonTests(unittest.TestCase):
    def output_for(self, scores):
        output = io.StringIO()
        with patch.object(DiagnosticModel, 'train_and_evaluate', return_value=scores), redirect_stdout(output):
            runpy.run_path(str(Path(__file__).with_name('tuning_analysis.py')), run_name='__main__')
        return output.getvalue()

    def test_lower_accuracy_is_reported_as_negative(self):
        output = self.output_for((0.8, 0.7))
        self.assertIn('-12.5%', output)
        self.assertNotIn('+-', output)

    def test_higher_accuracy_is_reported_as_positive(self):
        self.assertIn('+12.5%', self.output_for((0.8, 0.9)))

if __name__ == '__main__':
    unittest.main()
