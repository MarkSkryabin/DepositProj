import catboost
import shap
import pandas as pd
import io
from model_inference import catboost_frame


def build_force_plot(model, x, params):
    """
    Сохраняет force_plot для одного объекта в HTML.
    :param params: - все названия фичей (list)
    :param x: - входные данные модели (вектор-строка)
    :param model - предварительно обученная модель;
    :return: HTML-график без записи общих для пользователей файлов

    """
    # Преобразуем features_array в DataFrame, если это np.array
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=params)

    x = catboost_frame(model, x)
    pool_test = catboost.Pool(x, cat_features=model.get_cat_feature_indices(),
                              feature_names=params)

    shap_values = model.get_feature_importance(
        pool_test,
        type="ShapValues")
    # Индекс объекта (только 1 строка)
    i = 0
    # SHAP для конкретного объекта
    base_value = shap_values[i, -1]
    shap_values_object = shap_values[i, :-1]

    force_plot = shap.force_plot(
        base_value,
        shap_values_object,
        x.iloc[i],
        feature_names=params,
        show=False
    )

    output = io.StringIO()
    shap.save_html(output, force_plot)
    return output.getvalue()


def save_force_plot(model, x, cat_features, params, filename="shap_force.html"):
    """Backward-compatible file export for research scripts."""
    html = build_force_plot(model, x, params)
    with open(filename, 'w', encoding='utf-8') as output:
        output.write(html)
    return filename
