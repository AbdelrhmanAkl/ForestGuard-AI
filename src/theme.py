"""Design system: colors, global CSS and small HTML components."""
import streamlit as st

GREEN_DARK, GREEN, SAGE, MINT = "#183D32", "#2F6B50", "#A8BDA5", "#E8F0E8"
TEXT, MUTED, BORDER = "#24352D", "#718078", "#E1E8E0"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{font-family:'Inter',sans-serif;}}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{display:none;}}
.block-container {{max-width:1120px; padding:2rem 1.25rem 4rem;}}
h1,h2,h3,h4 {{color:{GREEN_DARK}; letter-spacing:-0.02em;}}
.fg-brand {{font-weight:700; font-size:1.15rem; color:{GREEN_DARK}; padding-top:.35rem;}}
.fg-brand span {{color:{GREEN};}}
.fg-hero {{background:linear-gradient(135deg,{GREEN_DARK},{GREEN}); color:#F7F8F4; border-radius:20px; padding:3.2rem 2.4rem; margin:1rem 0 1.5rem;}}
.fg-hero h1 {{color:#F7F8F4; font-size:clamp(1.9rem,4.5vw,3rem); line-height:1.1; margin:0 0 .8rem; padding:0;}}
.fg-hero p {{color:#DCE8DC; font-size:1.1rem; max-width:560px; margin:0;}}
.fg-eyebrow {{text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:600; color:{SAGE}; margin-bottom:.8rem;}}
.fg-grid {{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin:.5rem 0 1.5rem;}}
.fg-kpi, .fg-card {{background:#fff; border:1px solid {BORDER}; border-radius:14px; padding:1.1rem 1.25rem;}}
.fg-kpi .v {{font-size:2rem; font-weight:700; color:{GREEN_DARK}; line-height:1.1;}}
.fg-kpi .l {{font-size:.85rem; color:{MUTED}; margin-top:.3rem;}}
.fg-card h4 {{margin:0 0 .4rem; font-size:1rem;}}
.fg-card p {{margin:0; color:{MUTED}; font-size:.92rem; line-height:1.5;}}
.fg-step {{font-size:.75rem; font-weight:700; color:{GREEN}; letter-spacing:.08em; margin-bottom:.4rem;}}
.fg-result {{background:#fff; border:1px solid {BORDER}; border-left:5px solid {GREEN}; border-radius:14px; padding:1.5rem 1.75rem; margin:1rem 0;}}
.fg-result .k {{font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; color:{MUTED};}}
.fg-result .n {{font-size:2rem; font-weight:700; color:{GREEN_DARK}; margin:.2rem 0;}}
.fg-bar {{height:8px; background:{MINT}; border-radius:99px; overflow:hidden; margin-top:.5rem;}}
.fg-bar div {{height:100%; background:{GREEN}; border-radius:99px;}}
.fg-muted {{color:{MUTED}; font-size:.92rem;}}
.stButton>button, .stLinkButton>a {{border-radius:10px; font-weight:600;}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def section(title, caption=None):
    st.markdown(f"## {title}")
    if caption:
        st.markdown(f"<p class='fg-muted'>{caption}</p>", unsafe_allow_html=True)


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"<div class='fg-hero'><div class='fg-eyebrow'>{eyebrow}</div>"
        f"<h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def kpi_row(items):
    cells = "".join(f"<div class='fg-kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in items)
    st.markdown(f"<div class='fg-grid'>{cells}</div>", unsafe_allow_html=True)


def card_row(items):
    cells = "".join(
        f"<div class='fg-card'>{'<div class=fg-step>' + tag + '</div>' if tag else ''}<h4>{h}</h4><p>{p}</p></div>"
        for tag, h, p in items
    )
    st.markdown(f"<div class='fg-grid'>{cells}</div>", unsafe_allow_html=True)


def style_figure(fig, height=420):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT), margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig
