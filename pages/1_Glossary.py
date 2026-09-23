import streamlit as st

from glossary import GLOSSARY

st.set_page_config(page_title="Glossary", page_icon="📖", layout="wide")
st.title("📖 Glossary")
q = st.text_input("Search", placeholder="e.g. PEG, debt, fee…").lower()

for name, technical, simple in GLOSSARY:
    if q and q not in name.lower() and q not in technical.lower():
        continue
    with st.expander(name):
        left, right = st.columns(2)
        left.markdown("**🎓 Technical**")
        left.write(technical)
        right.markdown("**🧒 Like I'm 5**")
        right.write(simple)
