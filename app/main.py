import streamlit as st
import requests

def get_sagemaker_endpoint():
    try:
        response = requests.get("https://pokeapi.co/api/v2/pokemon/ditto")
        response.raise_for_status()  # Raise an error for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching SageMaker endpoint: {e}")
        return None

def main():
    st.title("My Streamlit App")
    st.write("Welcome to your Streamlit app base!")

    endpoint_data = get_sagemaker_endpoint()
    if endpoint_data:
        st.write("SageMaker Endpoint Data:")
        st.json(endpoint_data)

if __name__ == "__main__":
    main()