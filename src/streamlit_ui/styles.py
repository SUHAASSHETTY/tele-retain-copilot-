"""Small CSS layer on top of the Streamlit theme (.streamlit/config.toml). Only what the theme cannot do."""

import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stSidebar"] { border-right: 1px solid #e4e7ec; }
[data-testid="stMetric"] { background: #fff; }
[data-testid="stMetricLabel"] p { color: #475467; font-weight: 600; }

.rc-hero h1 { font-size: 2.05rem; margin: 0 0 .2rem 0; padding: 0; }
.rc-hero p { color: #475467; font-size: 1.05rem; margin: 0; }
.rc-eyebrow { color: #3b5bdb; font-weight: 700; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
  margin-bottom: .35rem; }
.rc-section { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #667085;
  margin: 0 0 .5rem 0; }
.rc-muted { color: #667085; font-size: .88rem; }

.rc-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: #f2f4f7; color: #344054; margin-right: 4px; white-space: nowrap; }
.rc-pill.high { background: #fef3f2; color: #b42318; }
.rc-pill.medium { background: #fffaeb; color: #b54708; }
.rc-pill.low { background: #ecfdf3; color: #067647; }
.rc-pill.info { background: #eef4ff; color: #3538cd; }

.rc-flow { display: flex; align-items: stretch; gap: .5rem; flex-wrap: wrap; margin: .25rem 0 .5rem 0; }
.rc-flow .step { flex: 1 1 150px; border: 1px solid #e4e7ec; border-radius: 12px; padding: .85rem .9rem; background: #fff; }
.rc-flow .step .n { width: 26px; height: 26px; border-radius: 999px; background: #eef2ff; color: #3b5bdb; font-weight: 700;
  display: flex; align-items: center; justify-content: center; font-size: .8rem; margin-bottom: .45rem; }
.rc-flow .step b { display: block; font-size: .95rem; color: #101828; }
.rc-flow .step span { display: block; color: #667085; font-size: .8rem; margin-top: .2rem; line-height: 1.35; }
.rc-flow .arrow { align-self: center; color: #98a2b3; font-size: 1.1rem; }
@media (max-width: 900px) { .rc-flow .arrow { display: none; } }

.rc-action { font-size: 1.3rem; font-weight: 700; color: #101828; line-height: 1.3; margin: .1rem 0 .4rem 0; }
.rc-kv { display: grid; grid-template-columns: 7.5rem 1fr; gap: .3rem .75rem; font-size: .9rem; }
.rc-kv .k { color: #667085; }
.rc-kv.stacked { grid-template-columns: 1fr; gap: .1rem; }
.rc-kv.stacked .v { margin-bottom: .55rem; }
.rc-kv .v { color: #101828; font-weight: 500; overflow-wrap: anywhere; }
.rc-factor { display: flex; gap: .6rem; padding: .4rem 0; border-bottom: 1px dashed #eaecf0; font-size: .9rem; }
.rc-factor:last-child { border-bottom: 0; }
.rc-dot { width: 9px; height: 9px; border-radius: 999px; margin-top: .42rem; flex: none; background: #98a2b3; }
.rc-dot.high { background: #d92d20; } .rc-dot.medium { background: #f79009; } .rc-dot.low { background: #12b76a; }
.rc-factor small { display: block; color: #667085; }
.rc-cite { font-size: .86rem; padding: .25rem 0; }
.rc-cite code { color: #3538cd; background: #eef2ff; }
</style>
"""


def inject() -> None:
    st.html(CSS)
