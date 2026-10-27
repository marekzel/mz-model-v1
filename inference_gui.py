
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

col_space, col_clear = st.columns([8, 2])
with col_clear:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if user_query := st.chat_input("Type a message..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        history_pool = st.session_state.messages[:-1]
        recent_messages = history_pool[-2:]

        history_str = ""
        for msg in recent_messages:
            role = msg["role"]
            history_str += f"<|{role}|>\n{msg['content']}\n"

        formatted_prompt = (
            history_str
            + f"<|user|>\n{user_query}\n"
            + "<|assistant|>\n"
        )

        input_ids = tokenizer.encode(formatted_prompt).tolist()
        x = torch.tensor(
            [input_ids], dtype=torch.long, device=device
        )

        generated_tokens = []
        eos_id = tokenizer.tokenizer.eos_token_id

        temperature = 0.4
        top_k = 30
        repetition_penalty = 1.3
        max_new_tokens = 120

        final_response = ""

        use_bf16 = (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                x_input = x[:, -512:]

                if use_bf16:
                    with torch.amp.autocast(
                        "cuda", dtype=torch.bfloat16
                    ):
                        logits, _ = model(x_input)
                else:
                    logits, _ = model(x_input)

                next_token_logits = (
                    logits[:, -1, :] / temperature
                )

                for token_id in set(generated_tokens):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

                if top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))
                    topk_values, _ = torch.topk(
                        next_token_logits, k
                    )
                    threshold = topk_values[..., -1, None]
                    next_token_logits[
                        next_token_logits < threshold
                    ] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()

                if token_id == eos_id:
                    break

                generated_tokens.append(token_id)
                x = torch.cat((x, next_token), dim=1)

                current_text = tokenizer.decode(generated_tokens)

                match = re.search(
                    r"<\|(user|assistant)\|>", current_text
                )
                if match:
                    current_text = current_text[:match.start()]
                    break

                response_placeholder.markdown(current_text + "▌")

        final_response = tokenizer.decode(generated_tokens).strip()

        match = re.search(
            r"<\|(user|assistant)\|>", final_response
        )
        if match:
            final_response = final_response[:match.start()].strip()

        if not final_response:
            final_response = "[Empty response]"

        response_placeholder.markdown(final_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": final_response}
    )

