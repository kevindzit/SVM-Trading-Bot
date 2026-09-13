import glob
import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backtest(name, relative_path):
    real_glob = glob.glob
    model_lookups = []

    def model_paths(pattern, *args, **kwargs):
        if str(pattern).endswith('*.joblib'):
            model_lookups.append(pattern)
            return ['fixture_MODEL.joblib', 'fixture_SCALER.joblib']
        return real_glob(pattern, *args, **kwargs)

    # let the old scripts import too, so reverting the fix tests the actual features
    with patch('glob.glob', side_effect=model_paths), patch('joblib.load') as load:
        module = load_module(name, relative_path)
        if load.called:
            raise AssertionError('Importing a backtest must not load saved models')
    return module, model_lookups


def sample_prices():
    steps = np.arange(120, dtype=float)
    close = 200 + 0.7 * steps + 3 * np.sin(steps / 5) + np.cos(steps / 8)
    return pd.DataFrame({
        'Open': close - 0.5,
        'High': close + 2,
        'Low': close - 2,
        'Close': close,
        'Volume': 1000 + 5 * steps,
    }, index=pd.date_range('2025-01-01', periods=len(steps), freq='h'))


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_module('svm_training', 'SVM_Bot/SVM_Model_Gen.py')
        cls.backtests = {}
        for name in ('Backtesting', 'Backtest_MoreTrades'):
            cls.backtests[name] = load_backtest(name, 'SVM_backtest/' + name + '.py')

    def setUp(self):
        self.prices = sample_prices()
        with redirect_stdout(io.StringIO()):
            self.training_features = self.training.calculate_features(self.prices.copy())

    def test_backtests_match_all_training_features(self):
        expected = self.training_features.dropna()
        self.assertFalse(expected.empty)
        self.assertEqual(len(expected.columns), 10)
        for name, (module, _) in self.backtests.items():
            with self.subTest(script=name):
                self.assertEqual(module.SvmStrategy.feature_names, expected.columns.tolist())
                actual = module.add_features(self.prices)[module.SvmStrategy.feature_names]
                pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)

    def test_band_width_is_in_price_units(self):
        # upper minus lower is four standard deviations for bands at +/- two
        expected = 4 * self.prices['Close'].rolling(20).std(ddof=0)
        pd.testing.assert_series_equal(
            self.training_features['BB_Width_20_2'], expected,
            check_names=False, rtol=1e-12, atol=1e-12,
        )
        for name, (module, _) in self.backtests.items():
            with self.subTest(script=name):
                actual = module.add_features(self.prices)['BB_Width_20_2']
                pd.testing.assert_series_equal(
                    actual, expected.loc[actual.index],
                    check_names=False, rtol=1e-12, atol=1e-12,
                )

    def test_trade_price_scaling_preserves_model_features(self):
        for name, (module, _) in self.backtests.items():
            with self.subTest(script=name):
                features = module.add_features(self.prices)
                scaled = module.scale_prices(features, module.BTC_PER_SHARE)
                columns = module.SvmStrategy.feature_names
                pd.testing.assert_frame_equal(scaled[columns], features[columns])
                np.testing.assert_allclose(
                    scaled['Close'], features['Close'] * module.BTC_PER_SHARE,
                )
                np.testing.assert_allclose(
                    scaled['Volume'], features['Volume'] / module.BTC_PER_SHARE,
                )

    def test_import_does_not_search_for_local_models(self):
        for name, (_, model_lookups) in self.backtests.items():
            with self.subTest(script=name):
                self.assertEqual(model_lookups, [])


if __name__ == '__main__':
    unittest.main()
