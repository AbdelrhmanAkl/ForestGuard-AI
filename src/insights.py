"""Loaders and charts for the existing evaluation reports (no data is modified)."""
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.theme import GREEN, GREEN_DARK, SAGE, style_figure
from src.utils import get_cover_type

REPORTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"


def load_report(name, **kwargs):
    """Read a report CSV; return None when it is unavailable."""
    path = REPORTS_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def class_performance_chart(report):
    rows = report.loc[report.iloc[:, 0].astype(str).str.fullmatch(r"\d+")].copy()
    rows["Cover Type"] = rows.iloc[:, 0].astype(int).map(lambda c: get_cover_type(c)["name"])
    rows["F1-score"] = rows["f1-score"]
    fig = px.bar(rows, x="Cover Type", y="F1-score", text=rows["F1-score"].map("{:.3f}".format),
                 color_discrete_sequence=[GREEN])
    fig.update_yaxes(range=[0, 1.05])
    return style_figure(fig, 380), rows


def comparison_chart(comparison):
    long = comparison.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_map={"Accuracy": GREEN_DARK, "Macro F1": SAGE})
    fig.update_yaxes(range=[0.8, 1.0])
    fig.update_layout(legend=dict(orientation="h", y=1.12, title=None))
    return style_figure(fig, 400).update_layout(showlegend=True)


def importance_chart(importance, top_n=15):
    top = importance.head(top_n).iloc[::-1]
    fig = px.bar(top, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[GREEN])
    return style_figure(fig, 30 * top_n + 80)


def confusion_chart(matrix, normalize):
    data = matrix.set_index(matrix.columns[0])
    labels = [get_cover_type(i + 1)["name"] for i in range(len(data))]
    values = data.div(data.sum(axis=1), axis=0) * 100 if normalize else data
    fig = px.imshow(values.values, x=labels, y=labels, text_auto=".1f" if normalize else ",",
                    color_continuous_scale=["#F7F8F4", GREEN_DARK], aspect="auto",
                    labels=dict(x="Predicted", y="Actual", color="%" if normalize else "Count"))
    fig.update_xaxes(side="bottom", tickangle=-35)
    return style_figure(fig, 520).update_layout(coloraxis_showscale=False)
