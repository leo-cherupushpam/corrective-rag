"""
utils.py
========
Utility functions for CRAG UI, including chart generation and data formatting.

Provides:
  - Chart generation (Plotly-based)
  - Data formatting helpers
  - HTML component builders
  - Trace visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
from crag import QueryTrace, GradeTrace
from styles import COLORS, get_relevance_color, get_confidence_label, get_relevance_label


# ============================================================================
# Chart Generation
# ============================================================================

def chart_relevance_scores(grades: List[GradeTrace], query: str = "") -> go.Figure:
    """
    Create a horizontal bar chart of document relevance scores.

    Args:
        grades: List of GradeTrace objects with relevance scores
        query: Query text for context

    Returns:
        Plotly figure
    """
    if not grades:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(text="No documents graded", xref="paper", yref="paper")
        return fig

    # Prepare data
    labels = []
    scores = []
    colors_list = []

    for i, grade in enumerate(grades):
        # Truncate preview to 40 chars
        preview = (grade.document_preview[:40] + "...") if len(grade.document_preview) > 40 else grade.document_preview
        labels.append(f"Doc {i+1}: {preview}")
        scores.append(grade.score)
        colors_list.append(get_relevance_color(grade.score))

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=scores,
        orientation='h',
        marker=dict(color=colors_list, line=dict(color=COLORS["border"], width=1)),
        text=[f"{s:.0%}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Relevance: %{x:.0%}<extra></extra>",
    ))

    # Add threshold line at 0.5 (typical relevance threshold)
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Threshold", annotation_position="top")

    fig.update_layout(
        title=f"Document Relevance Evaluation" + (f" for '{query[:30]}...'" if query else ""),
        xaxis_title="Relevance Score",
        yaxis_title="Documents",
        height=300 + (len(grades) * 30),
        margin=dict(l=300, r=100),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_dark"]),
    )

    fig.update_xaxes(range=[0, 1], showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(autorange="reversed")

    return fig


def chart_cost_breakdown(baseline_cost: float, crag_cost: float,
                         breakdown: Dict[str, float]) -> go.Figure:
    """
    Create a cost comparison and breakdown chart.

    Args:
        baseline_cost: Cost for baseline RAG
        crag_cost: Cost for CRAG
        breakdown: Dict of component -> cost (for CRAG)

    Returns:
        Plotly figure with 2 subplots: comparison bars + breakdown pie
    """
    from plotly.subplots import make_subplots

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Cost Comparison", "CRAG Cost Breakdown"),
    )

    # Bar chart: Baseline vs CRAG
    fig.add_trace(
        go.Bar(
            x=["Baseline RAG", "CRAG"],
            y=[baseline_cost, crag_cost],
            marker=dict(color=[COLORS["info"], COLORS["warning"]]),
            text=[f"${baseline_cost:.4f}", f"${crag_cost:.4f}"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Cost: $%{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Pie chart: Cost breakdown
    if breakdown:
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        colors_list = [COLORS["info"], COLORS["warning"], COLORS["error"], COLORS["success"]][:len(labels)]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors_list),
                textposition="inside",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Cost: $%{value:.4f} (%{percent})<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title_text="Query Cost Analysis",
        height=400,
        margin=dict(b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_dark"]),
        showlegend=True,
    )

    return fig


def chart_confidence_calibration(queries_data: List[Dict]) -> go.Figure:
    """
    Create a confidence calibration scatter plot.

    Args:
        queries_data: List of dicts with 'confidence' and 'correct' keys

    Returns:
        Plotly figure
    """
    if not queries_data or len(queries_data) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 queries to show calibration", xref="paper", yref="paper")
        return fig

    confidences = [q.get("confidence", 0) for q in queries_data]
    correctness = [1 if q.get("correct", False) else 0 for q in queries_data]

    fig = go.Figure()

    # Plot correct answers
    correct_mask = [c == 1 for c in correctness]
    fig.add_trace(go.Scatter(
        x=[c for c, m in zip(confidences, correct_mask) if m],
        y=[1] * sum(correct_mask),
        mode='markers',
        marker=dict(size=10, color=COLORS["success"], symbol="circle"),
        name="Correct",
        hovertemplate="Confidence: %{x:.0%}<br>Outcome: Correct<extra></extra>",
    ))

    # Plot incorrect answers (hallucinations)
    incorrect_mask = [c == 0 for c in correctness]
    fig.add_trace(go.Scatter(
        x=[c for c, m in zip(confidences, incorrect_mask) if m],
        y=[0] * sum(incorrect_mask),
        mode='markers',
        marker=dict(size=10, color=COLORS["error"], symbol="x"),
        name="Hallucination",
        hovertemplate="Confidence: %{x:.0%}<br>Outcome: Hallucination<extra></extra>",
    ))

    # Add diagonal line (perfect calibration)
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=COLORS["text_light"], width=2, dash="dash"),
                  layer="below")

    fig.update_layout(
        title="Confidence Calibration (Perfect: points on diagonal)",
        xaxis_title="Predicted Confidence",
        yaxis_title="Actual Correctness",
        height=500,
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_dark"]),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-0.1, 1.1]),
    )

    return fig


def chart_hallucination_metrics(baseline_rate: float, crag_rate: float) -> go.Figure:
    """
    Create a gauge chart showing hallucination reduction.

    Args:
        baseline_rate: Baseline hallucination rate (0.0-1.0)
        crag_rate: CRAG hallucination rate (0.0-1.0)

    Returns:
        Plotly figure with 2 gauges
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Baseline RAG", "CRAG"),
        horizontal_spacing=0.2,
    )

    # Baseline gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=baseline_rate * 100,
        title={"text": "Hallucination Rate"},
        delta={"reference": 15},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["error"]},
            "steps": [
                {"range": [0, 5], "color": COLORS["success_light"]},
                {"range": [5, 15], "color": COLORS["warning_light"]},
                {"range": [15, 100], "color": COLORS["error_light"]},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 20,
            },
        },
        number={"suffix": "%"},
    ), row=1, col=1)

    # CRAG gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=crag_rate * 100,
        title={"text": "Hallucination Rate"},
        delta={"reference": 15},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["success"]},
            "steps": [
                {"range": [0, 5], "color": COLORS["success_light"]},
                {"range": [5, 15], "color": COLORS["warning_light"]},
                {"range": [15, 100], "color": COLORS["error_light"]},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 20,
            },
        },
        number={"suffix": "%"},
    ), row=1, col=2)

    fig.update_layout(
        title_text=f"Hallucination Reduction: {(baseline_rate - crag_rate) * 100:.0f}%",
        height=400,
        margin=dict(b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_dark"]),
    )

    return fig


# ============================================================================
# Data Formatting Helpers
# ============================================================================

def format_cost(cost_usd: float) -> str:
    """Format cost as USD string."""
    if cost_usd < 0.0001:
        return f"${cost_usd * 1_000_000:.2f}μ"
    elif cost_usd < 0.01:
        return f"${cost_usd * 1000:.3f}m"
    else:
        return f"${cost_usd:.4f}"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence:.0%}"


def get_confidence_emoji(confidence: float) -> str:
    """Get confidence emoji based on score."""
    label, _ = get_confidence_label(confidence)
    return label.split()[0]  # Just the emoji


# ============================================================================
# Component Builders
# ============================================================================

def make_correction_flow(corrections: List) -> str:
    """
    Create an HTML visualization of the correction flow.

    Args:
        corrections: List of CorrectionTrace objects

    Returns:
        HTML string
    """
    if not corrections:
        return "<p>No corrections needed ✅</p>"

    html = """
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 12px;">
        <div style="font-weight: bold; color: #1F2937;">Query</div>
    """

    for i, correction in enumerate(corrections):
        # Strategy box
        html += f"""
        <div style="
            background-color: #FDE8D0;
            border: 1px solid #F18F01;
            border-radius: 6px;
            padding: 6px 10px;
            color: #1F2937;
            font-weight: 600;
        ">
            {correction.strategy}
        </div>
        """

        # Arrow
        if i < len(corrections) - 1:
            html += '<div style="color: #64748B; font-weight: bold;">→</div>'

    html += """
        <div style="font-weight: bold; color: #06A77D;">✓ Found</div>
    </div>
    """

    return html


def make_multi_hop_flow(hops: List) -> str:
    """
    Create an HTML visualization of multi-hop retrieval.

    Args:
        hops: List of MultiHopTrace objects

    Returns:
        HTML string
    """
    if not hops:
        return "<p>Single-hop retrieval</p>"

    html = """
    <div style="display: flex; flex-direction: column; gap: 12px;">
    """

    for hop in hops:
        html += f"""
        <div style="
            background-color: {COLORS['info_light']};
            border: 1px solid {COLORS['info']};
            border-radius: 6px;
            padding: 12px;
            color: {COLORS['text_dark']};
        ">
            <div style="font-weight: 600; margin-bottom: 6px;">
                🔍 Hop {hop.hop_number}: {hop.bridge_entity}
            </div>
            <div style="font-size: 12px; color: {COLORS['text_light']};">
                Query: {hop.bridge_query}<br>
                Retrieved: {hop.docs_retrieved} docs → Passed: {hop.docs_passed_grade} docs
            </div>
        </div>
        """

    html += "</div>"
    return html
