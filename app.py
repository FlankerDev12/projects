import streamlit as s
import requests

s.set_page_config(page_title="Quntum Mechanics Bot", page_icon="⚛️")
s.title("⚛️ Quantum Mechanics Assistant")
s.markdown("Ask me anything about Quantum Mechanic or Anything related to it!")

user_input = s.text_input("Enter your Question here:")
if s.button("ASK"):
    if user_input.strip() == "":
        s.warning("Please enter a valid question.")
    else:
        try:
            response = requests.get(
                "http://127.0.0.1:8000/troubleshoot",
                params={"issue": user_input}
            )
            data=response.json()
            if "troubleshoot_guide" in data:
                s.markdown(f"**Answer:** {data['troubleshoot_guide']}")
            else:
                s.error("Unexpected response format from the server.")

        except Exception as e:
            s.error(f"An error occurred: {e}")
