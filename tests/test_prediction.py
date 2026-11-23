import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from catboost import CatBoostClassifier
from streamlit.testing.v1 import AppTest

from model_inference import load_model_bundle, predict_ensemble
from shap_file_stremlit import build_force_plot, save_force_plot


ROOT = Path(__file__).resolve().parents[1]


class BaseModel:
    def __init__(self, kind, probability):
        self.kind, self.probability = kind, probability

    def get_cat_feature_indices(self):
        return [1]

    def predict_proba(self, frame):
        assert list(frame.columns) == ['amount', 'job']
        if self.kind == 'LightGBM':
            assert isinstance(frame['job'].dtype, pd.CategoricalDtype)
        else:
            assert frame['job'].dtype == object
        return np.tile([1-self.probability, self.probability], (len(frame), 1))


class MetaModel:
    def predict_proba(self, data):
        np.testing.assert_allclose(data, [[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]])
        return np.tile([0.6, 0.4], (len(data), 1))


def test_ensemble_does_not_change_input_or_mix_model_dtypes():
    frame = pd.DataFrame({'amount': [1.0, 2.0], 'job': ['management', 'admin.']})
    original = frame.copy(deep=True)
    models = [BaseModel(kind, probability)
              for kind, probability in [('TabM', 0.2), ('CatBoost', 0.4), ('LightGBM', 0.6)]
              for _ in range(5)]
    np.testing.assert_allclose(predict_ensemble(models, MetaModel(), frame), [0.4, 0.4])
    assert_frame_equal(frame, original)


def test_shap_export_handles_windows_encoding_and_category_dtype(tmp_path):
    train = pd.DataFrame({'amount': [1, 2, 3, 4], 'job': ['работник', 'менеджер']*2})
    model = CatBoostClassifier(iterations=2, verbose=False, cat_features=['job'],
                               allow_writing_files=False)
    model.fit(train, [0, 1, 0, 1])
    row = train.iloc[:1].copy()
    row['job'] = row['job'].astype('category')
    before = row.copy(deep=True)
    # SHAP's JavaScript contains characters that cannot be written using the
    # default Windows cp1251 codec. Production HTML generation must stay in memory.
    real_open = io.open

    def windows_open(file, mode='r', *args, **kwargs):
        if 'w' in mode and 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'cp1251'
        return real_open(file, mode, *args, **kwargs)

    with patch('io.open', windows_open):
        html = build_force_plot(model, row, list(row.columns))
    assert '<html>' in html and 'featureNames' in html
    assert_frame_equal(row, before)
    target = tmp_path / 'plot.html'
    save_force_plot(model, row, ['job'], list(row.columns), target)
    assert '<html>' in target.read_text(encoding='utf-8')


def test_single_prediction_and_session_analytics_with_saved_models():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not at.exception
    job = next(widget for widget in at.selectbox if widget.label == 'Тип занятости (job)')
    assert 'менеджер / руководитель' in job.options
    job.select('менеджер / руководитель').run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state['shap_plots']) == 5
    assert any('Клиент' in item.value and '%' in item.value for item in at.markdown)
    with patch('streamlit_option_menu.option_menu', return_value='Аналитика'):
        at.run()
    assert not at.exception
    assert len(at.get('iframe')) == 5


def test_analytics_before_prediction_has_no_stale_files():
    with patch('streamlit_option_menu.option_menu', return_value='Аналитика'):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not at.exception
    assert 'Сначала выполните одиночный прогноз' in at.info[0].value


def test_batch_prediction_with_saved_models_from_another_working_directory(monkeypatch, tmp_path):
    from train_functions import create_new_features, preprocessing

    raw = pd.read_csv(ROOT / 'test.csv', nrows=2)
    monkeypatch.chdir(tmp_path)
    frame = preprocessing(create_new_features(raw))
    original = frame.copy(deep=True)
    models, meta_model = load_model_bundle('iteration_14')
    probability = predict_ensemble(models, meta_model, frame)
    assert probability.shape == (2,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert_frame_equal(frame, original)
