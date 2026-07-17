"""Model/tokenizer loading and architecture-agnostic module access."""
import torch


def load_tokenizer(mcfg: dict):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        mcfg["hf_id"], trust_remote_code=mcfg.get("trust_remote_code", False)
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(mcfg: dict, attn_impl: str | None = "eager", for_training: bool = False):
    """attn_impl='eager' is required whenever output_attentions=True is used."""
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, mcfg.get("dtype", "float16"))
    kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl
    if not for_training:
        kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], **kwargs)
    except (ValueError, TypeError):
        # some remote-code architectures reject attn_implementation
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], **kwargs)
    if for_training and torch.cuda.is_available():
        model = model.to("cuda")
    if not for_training:
        model.eval()
    return model


def resolve_module(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def decoder_layers(model, mcfg: dict):
    return resolve_module(model, mcfg.get("layers_path", "model.layers"))


def mlp_module(layer, mcfg: dict, key: str):
    """key in {gate, up, act, down} -> the corresponding submodule of a decoder layer."""
    return resolve_module(layer, mcfg["module_map"][key])


def token_str(tok, token_id: int) -> str:
    """Human-facing string for a single vocab id (keeps leading-space semantics)."""
    try:
        return tok.convert_tokens_to_string(tok.convert_ids_to_tokens([token_id]))
    except Exception:
        return tok.decode([token_id])
