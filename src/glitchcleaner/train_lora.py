"""Train the gated LoRA branches (paper: r=4, alpha=4, MLP gate+data of key layers only).

The 'gate' is realized at inference time by enabling/disabling the adapter
(lambda=1 <=> adapter active), exactly mirroring the paper's lambda switch.
"""
import torch

from ..common.model_utils import mlp_module, decoder_layers


def target_module_names(model, mcfg: dict, targets: list[str]) -> list[str]:
    """Fully-qualified module names of gate/up projections in key layers only."""
    names = []
    wanted = set()
    layers = decoder_layers(model, mcfg)
    for li in mcfg["key_layers"]:
        for t in targets:
            wanted.add(id(mlp_module(layers[li], mcfg, t)))
    for name, module in model.named_modules():
        if id(module) in wanted:
            names.append(name)
    return names


def train(model, tok, examples: list[dict], mcfg: dict, gc_cfg: dict, out_dir):
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    lcfg = gc_cfg["lora"]
    tcfg = gc_cfg["training"]

    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=target_module_names(model, mcfg, lcfg["target"]),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    def encode(ex):
        prompt_ids = tok(ex["prompt"], add_special_tokens=True).input_ids
        answer_ids = tok(ex["answer"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        input_ids = prompt_ids + answer_ids
        labels = list(input_ids)
        if gc_cfg["training"].get("mask_prompt_loss", True):
            labels[: len(prompt_ids)] = [-100] * len(prompt_ids)
        return {"input_ids": input_ids, "labels": labels}

    encoded = [encode(e) for e in examples]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids, labels, mask = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append([pad] * n + b["input_ids"])          # left pad
            labels.append([-100] * n + b["labels"])
            mask.append([0] * n + [1] * len(b["input_ids"]))
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    device = next(model.parameters()).device
    loader = DataLoader(encoded, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg["lr"]
    )
    accum = tcfg.get("gradient_accumulation", 1)

    model.train()
    for epoch in range(tcfg["epochs"]):
        total, steps = 0.0, 0
        optim.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            if (i + 1) % accum == 0:
                optim.step()
                optim.zero_grad()
            total += loss.item() * accum
            steps += 1
        print(f"epoch {epoch + 1}/{tcfg['epochs']}  mean_loss={total / max(steps, 1):.4f}")
    model.eval()

    model.save_pretrained(str(out_dir))
    print(f"adapter saved to {out_dir}")
    return model
