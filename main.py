import streamlit as st
from uuid import uuid4
from vllm import LLM, SamplingParams

class BKvLLM:
    def __init__(self,
                 model: str,
                 download_dir: str,
                 enforce_eager: bool = True,
                 quantization: str = "awq",
                 gpu_memory_utilization: float = 0.5,
                 max_model_len: int = 256,
                 trust_remote_code: bool = True,
                 max_tokens: int = 256,
                 min_tokens: int = 128,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 repetition_penalty: float = 1.2,
                 swap_space: int = 0):

        self.model = LLM(
            model=model,
            download_dir=download_dir,
            enforce_eager=enforce_eager,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            swap_space=swap_space,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

    def generate(self, prompt: str) -> str:
        response = self.model.generate(prompts=[prompt], sampling_params=self.sampling_params)

        if response and response[0].outputs:
            return response[0].outputs[0].text
        else:
            return "No output generated."

# Streamlit UI
st.title("AMBACHATBOT")
st.write("Enter a prompt below to generate a response.")

# Input form
with st.form("prompt_form"):
    user_input = st.text_area("Enter your prompt:", "")
    submit_button = st.form_submit_button("Generate")

if submit_button:
    # Load the model (adjust the model path if necessary)
    model_name = "./model_download_dir"  # Path to your downloaded model
    llm = BKvLLM(model=model_name, download_dir=model_name)

    # Generate response
    with st.spinner("Generating response..."):
        response = llm.generate(user_input)
    
    st.write("### Generated Response:")
    st.write(response)
