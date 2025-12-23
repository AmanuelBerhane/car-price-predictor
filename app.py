import streamlit as st
from model import get_model_and_df, recommend
from query_parser import parse_user_query
from pathlib import Path

st.set_page_config(page_title="Car Value Chat", layout="wide")
st.title("Car Value Assistant")

CSV_PATH = str((Path(__file__).resolve().parent  / "car_price_prediction_ (1).csv").resolve())

@st.cache_resource(show_spinner=False)
def _load():
    model, df = get_model_and_df(CSV_PATH)
    return model, df

if "history" not in st.session_state:
    st.session_state.history = []

st.write("Ask for cars by budget, brand, fuel, transmission, mileage, year, condition. Example: 'Under 20000, Toyota or Honda, automatic, after 2016, <80k miles'.")

TOP_N_DEFAULT = 5

user_msg = st.chat_input("Describe your needs (budget, brand, fuel, etc.)")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    constraints = parse_user_query(user_msg)
    model, df = _load()
    response_md = recommend(model, df, constraints, top_n=TOP_N_DEFAULT)
    st.session_state.history.append(("assistant", response_md))

for role, content in st.session_state.history:
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(content)
        else:
            st.write(content)
