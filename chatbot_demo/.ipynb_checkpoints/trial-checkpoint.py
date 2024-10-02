import streamlit as st

if 'x' not in st.session_state:
    st.session_state.x = False
if 'y' not in st.session_state:
    st.session_state.y = False

x = st.button("button x")
if x or st.session_state.x:
    st.write("x is on")
    st.session_state.x = True

    y = st.button("button y")
    if y or st.session_state.y:
        st.write("y is on")
        st.session_state.y = True
