import streamlit as st

from llm import ask

st.set_page_config(page_title="Ask AI", page_icon="🤖")
st.title("🤖 Ask AI")

a = st.session_state.get("analysis")
if not a:
    st.info("Open the main page and pick a stock or ETF first, then come back here.")
    st.stop()

st.caption(f"Discussing **{a['name']} ({a['ticker']})** · answers are educational, not advice")

# reset the chat when the asset changes
if st.session_state.get("chat_ticker") != a["ticker"]:
    st.session_state["chat"] = []
    st.session_state["chat_ticker"] = a["ticker"]

for m in st.session_state["chat"]:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Is this stock expensive? What are the main risks?"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = ask(prompt, a["summary"], st.session_state["chat"])
        st.write(reply)
    st.session_state["chat"] += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
