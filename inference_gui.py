
import streamlit as st
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from model import MzModelV1
from tokenizer import MzTokenizerWrapper
import re

st.set_page_config(page_title="MzModelV1 - Chat", page_icon="💬", layout="centered")
st.title("MzModelV1 Chat Interface")

BASE_PATH = Path(__file__).resolve().parent
MODEL_PATH = BASE_PATH / "mz-model-v1"
TOKENIZER_PATH = BASE_PATH / "tokenizer_llama2"
REPO_ID = "marekzel/mz-model-v1"


@st.cache_resource
def load_components():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if not (MODEL_PATH / "model.safetensors").exists() or not (MODEL_PATH / "config.json").exists():
        with st.spinner("Downloading MzModelV1 from Hugging Face..."):
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=MODEL_PATH,
                allow_patterns=["model.safetensors", "config.json"],
            )

    tokenizer = MzTokenizerWrapper(TOKENIZER_PATH)
    model = MzModelV1.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    return model, tokenizer, device


model, tokenizer, device = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Type a message..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        formatted_prompt = f"<|user|>\n{user_query}\n<|assistant|>\n"
        input_ids = tokenizer.encode(formatted_prompt).tolist()
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated_tokens = []
        eos_id = tokenizer.tokenizer.eos_token_id
        temperature = 0.6
        top_k = 30
        repetition_penalty = 1.1

        with torch.no_grad():
            for _ in range(120):
                x_input = x[:, -512:] if x.size(1) > 512 else x

                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(x_input)
                else:
                    logits, _ = model(x_input)

                next_token_logits = logits[:, -1, :] / temperature

                for token_id in set(generated_tokens):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

                if top_k > 0:
                    topk_values, _ = torch.topk(next_token_logits, top_k)
                    indices_to_remove = next_token_logits < topk_values[..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                if token_id == eos_id:
                    break

                generated_tokens.append(token_id)
                x = torch.cat((x, next_token), dim=1)

                current_text = tokenizer.decode(generated_tokens)

                stop_tags = ["<|user|>", "<|assistant|>"]

                if any(tag in current_text for tag in stop_tags):
                    break

                response_placeholder.markdown(current_text + "▌")

                final_response = tokenizer.decode(generated_tokens).strip()

                final_response = re.sub(
                    r"<\|(?:user|assistant)(?:\|?)?$",
                    "",
                    final_response
                ).strip()

                for tag in ["<|user|>", "<|assistant|>", "### User:", "### Assistant:"]:
                    final_response = final_response.replace(tag, "")

                while True:
                    cleaned = final_response.strip()
                    matched = False

                    for prefix in ["Assistant:", "Assistant", "assistant:", "assistant"]:
                        if cleaned.startswith(prefix):
                            final_response = cleaned[len(prefix):].strip()
                            matched = True
                            break

                    if not matched:
                        break

                response_placeholder.markdown(final_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_response}
    )
