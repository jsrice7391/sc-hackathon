import streamlit as st
import boto3
import json
import os


def get_sagemaker_endpoint():
    client = boto3.client("sagemaker-runtime",
                           aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                           aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                           region_name="us-east-1")
    response = client.invoke_endpoint(
        EndpointName="pytorch-inference-2025-08-24-12-56-47-523",
        ContentType="application/json",
        Body=json.dumps({"key": "value"})
    )
    return json.loads(response["Body"].read())


def main():
    st.title("My Streamlit App")
    st.write("Welcome to your Streamlit app base!")

    endpoint_data = get_sagemaker_endpoint()
    if endpoint_data:
        st.write("SageMaker Endpoint Data:")
        st.json(endpoint_data)

if __name__ == "__main__":
    main()