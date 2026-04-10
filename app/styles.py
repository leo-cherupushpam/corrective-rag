"""
styles.py
=========
CRAG UI styling system with consistent colors, spacing, and typography.

Provides:
  - Color palette (success, warning, error, info, neutral)
  - CSS styling for custom components
  - Badge styling for confidence, cost, metrics
  - Dark mode support
"""

# ============================================================================
# Color Palette
# ============================================================================

COLORS = {
    # Primary actions and success states
    "success": "#06A77D",      # Green
    "success_light": "#D4F4E7",

    # Warning and caution states
    "warning": "#F18F01",      # Orange
    "warning_light": "#FDE8D0",

    # Error and negative states
    "error": "#D62828",        # Red
    "error_light": "#FADBD8",

    # Information and neutral states
    "info": "#2E86DE",         # Blue
    "info_light": "#D6EAF8",

    # Neutral colors
    "primary": "#1E3A8A",      # Dark blue
    "secondary": "#64748B",    # Slate
    "neutral": "#F0F0F0",      # Light gray
    "neutral_dark": "#E2E8F0", # Slightly darker gray
    "text_dark": "#1F2937",    # Almost black
    "text_light": "#6B7280",   # Medium gray
    "border": "#D1D5DB",       # Border gray

    # Status-specific
    "relevant": "#06A77D",     # Green (high relevance)
    "marginal": "#F18F01",     # Orange (medium relevance)
    "irrelevant": "#D62828",   # Red (low relevance)
}

# ============================================================================
# Spacing System (8px baseline)
# ============================================================================

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
}

# ============================================================================
# Typography
# ============================================================================

TYPOGRAPHY = {
    "h1": {"size": "28px", "weight": "700"},
    "h2": {"size": "24px", "weight": "700"},
    "h3": {"size": "20px", "weight": "700"},
    "h4": {"size": "18px", "weight": "600"},
    "body": {"size": "14px", "weight": "400"},
    "small": {"size": "12px", "weight": "400"},
}

# ============================================================================
# Custom CSS for Streamlit
# ============================================================================

def get_custom_css():
    """Return custom CSS for consistent styling across the app."""
    return f"""
    <style>
    /* Root color variables */
    :root {{
        --color-success: {COLORS['success']};
        --color-warning: {COLORS['warning']};
        --color-error: {COLORS['error']};
        --color-info: {COLORS['info']};
        --color-primary: {COLORS['primary']};
        --color-text-dark: {COLORS['text_dark']};
        --color-text-light: {COLORS['text_light']};
        --color-border: {COLORS['border']};

        --spacing-xs: {SPACING['xs']};
        --spacing-sm: {SPACING['sm']};
        --spacing-md: {SPACING['md']};
        --spacing-lg: {SPACING['lg']};
        --spacing-xl: {SPACING['xl']};
    }}

    /* Typography */
    h1 {{ font-size: {TYPOGRAPHY['h1']['size']}; font-weight: {TYPOGRAPHY['h1']['weight']}; }}
    h2 {{ font-size: {TYPOGRAPHY['h2']['size']}; font-weight: {TYPOGRAPHY['h2']['weight']}; }}
    h3 {{ font-size: {TYPOGRAPHY['h3']['size']}; font-weight: {TYPOGRAPHY['h3']['weight']}; }}

    /* Overall page padding */
    .main {{ padding: {SPACING['lg']}; }}

    /* Streamlit containers */
    [data-testid="stMetricValue"] {{
        font-size: 24px;
        font-weight: 700;
        color: {COLORS['text_dark']};
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 12px;
        color: {COLORS['text_light']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Custom containers */
    .metric-card {{
        background-color: {COLORS['neutral']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: {SPACING['md']};
        margin: {SPACING['sm']} 0;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .badge-success {{
        background-color: {COLORS['success_light']};
        color: {COLORS['success']};
        border: 1px solid {COLORS['success']};
    }}

    .badge-warning {{
        background-color: {COLORS['warning_light']};
        color: {COLORS['warning']};
        border: 1px solid {COLORS['warning']};
    }}

    .badge-error {{
        background-color: {COLORS['error_light']};
        color: {COLORS['error']};
        border: 1px solid {COLORS['error']};
    }}

    .badge-info {{
        background-color: {COLORS['info_light']};
        color: {COLORS['info']};
        border: 1px solid {COLORS['info']};
    }}

    /* Buttons and interactions */
    .stButton > button {{
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
    }}

    /* Code blocks */
    pre {{
        background-color: {COLORS['neutral']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: {SPACING['md']};
    }}

    /* Links */
    a {{
        color: {COLORS['info']};
        text-decoration: none;
    }}

    a:hover {{
        text-decoration: underline;
    }}

    /* Dividers */
    hr {{
        border-color: {COLORS['border']};
        margin: {SPACING['md']} 0;
    }}

    /* Info boxes */
    .stInfo {{
        background-color: {COLORS['info_light']};
        border-left: 4px solid {COLORS['info']};
    }}

    .stSuccess {{
        background-color: {COLORS['success_light']};
        border-left: 4px solid {COLORS['success']};
    }}

    .stWarning {{
        background-color: {COLORS['warning_light']};
        border-left: 4px solid {COLORS['warning']};
    }}

    .stError {{
        background-color: {COLORS['error_light']};
        border-left: 4px solid {COLORS['error']};
    }}

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --color-text-dark: #F3F4F6;
            --color-text-light: #D1D5DB;
            --color-border: #4B5563;
            --color-neutral: #1F2937;
        }}

        .metric-card {{
            background-color: #111827;
            border-color: #4B5563;
        }}

        pre {{
            background-color: #111827;
            border-color: #4B5563;
        }}
    }}
    </style>
    """


# ============================================================================
# Badge Styling Helpers
# ============================================================================

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence score (0.0-1.0)."""
    if confidence >= 0.7:
        return COLORS["success"]
    elif confidence >= 0.4:
        return COLORS["warning"]
    else:
        return COLORS["error"]


def get_relevance_color(score: float) -> str:
    """Get color based on relevance score (0.0-1.0)."""
    if score >= 0.7:
        return COLORS["relevant"]
    elif score >= 0.5:
        return COLORS["marginal"]
    else:
        return COLORS["irrelevant"]


def get_confidence_label(confidence: float) -> str:
    """Get label and emoji for confidence level."""
    if confidence >= 0.7:
        return "🟢 High", COLORS["success"]
    elif confidence >= 0.4:
        return "🟡 Medium", COLORS["warning"]
    else:
        return "🔴 Low", COLORS["error"]


def get_relevance_label(score: float) -> str:
    """Get label for relevance score."""
    if score >= 0.7:
        return "✅ Highly Relevant"
    elif score >= 0.5:
        return "⚠️ Somewhat Relevant"
    else:
        return "❌ Not Relevant"


# ============================================================================
# Component Builders
# ============================================================================

def make_confidence_badge(confidence: float, show_score: bool = True) -> str:
    """Create an HTML confidence badge."""
    label, color = get_confidence_label(confidence)
    score_text = f" ({confidence:.0%})" if show_score else ""

    return f"""
    <div style="
        display: inline-block;
        padding: 6px 12px;
        background-color: {color}20;
        color: {color};
        border: 1px solid {color};
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    ">
        {label}{score_text}
    </div>
    """


def make_metric_card(label: str, value: str, color: str = None, subtext: str = None) -> str:
    """Create an HTML metric card."""
    color = color or COLORS["primary"]

    html = f"""
    <div style="
        background-color: {COLORS['neutral']};
        border: 2px solid {color};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
    ">
        <div style="
            color: {COLORS['text_light']};
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        ">{label}</div>
        <div style="
            color: {color};
            font-size: 28px;
            font-weight: 700;
        ">{value}</div>
    """

    if subtext:
        html += f"""
        <div style="
            color: {COLORS['text_light']};
            font-size: 11px;
            margin-top: 8px;
        ">{subtext}</div>
        """

    html += "</div>"
    return html
