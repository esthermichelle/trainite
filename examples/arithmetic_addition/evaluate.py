import argparse
from pathlib import Path

import torch

from config import ProjectConfig
from utils import build_dataset, build_model, instantiate, load_config

BATCH_SIZE = 64


@torch.no_grad()
def generate_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int, max_new_tokens: int):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated, attention_mask=attention_mask)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if eos_id is not None:
            already_ended = generated[:, -1:].eq(eos_id)
            next_token = torch.where(already_ended, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token], dim=-1)
        next_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        if eos_id is not None:
            already_ended = generated[:, -2:-1].eq(eos_id)
            next_mask = torch.where(already_ended, torch.zeros_like(next_mask), next_mask)
        attention_mask = torch.cat([attention_mask, next_mask], dim=-1)
        if generated[:, -1].eq(eos_id).all():
            break
    return generated


def extract_completion(ids: list[int], sep_id: int, eos_id: int, decode) -> str:
    if sep_id in ids:
        completion = ids[ids.index(sep_id) + 1 :]
    else:
        completion = ids
    if eos_id is not None and eos_id in completion:
        completion = completion[: completion.index(eos_id)]
    return decode(completion, skip_special_tokens=True)


def left_pad(items, tokenizer, device):
    seqs = [item.eval_input_ids.tolist() for item in items]
    max_len = max(len(seq) for seq in seqs)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    input_ids = torch.tensor([[pad_id] * (max_len - len(seq)) + seq for seq in seqs], device=device)
    attention_mask = torch.tensor([[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs], device=device)
    return input_ids, attention_mask


def evaluate_split(
    model, tokenizer, data_config, device, max_new_tokens: int
) -> tuple[float, int, list[tuple[str, str, str]]]:
    dataset = build_dataset(data_config.dataset, data_config.transform, tokenizer)
    wrong: list[tuple[str, str, str]] = []
    correct = 0

    for start in range(0, len(dataset), BATCH_SIZE):
        items = [dataset[i] for i in range(start, min(start + BATCH_SIZE, len(dataset)))]
        input_ids, attention_mask = left_pad(items, tokenizer, device)
        generated = generate_batch(
            model,
            input_ids,
            attention_mask,
            eos_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
        )
        for item, gen in zip(items, generated.tolist()):
            prediction = extract_completion(gen, tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.decode)
            if prediction == item.target:
                correct += 1
            elif len(wrong) < 10:
                wrong.append((item.source, str(item.target), prediction))

    n = len(dataset)
    return correct / n, n, wrong


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to best.pt (defaults to newest run)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    config: ProjectConfig = load_config(project_dir / args.config, ProjectConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = instantiate(config.preprocessor)
    model = build_model(config.model, device, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        run_dirs = sorted(
            (project_dir / config.output.root / config.output.run_name).glob("*/best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not run_dirs:
            raise FileNotFoundError("No best.pt found; train first with: python main.py config.yaml")
        checkpoint_path = run_dirs[-1]
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Evaluating checkpoint: {checkpoint_path}")

    max_new_tokens = config.trainer.max_inference_new_tokens
    for name, split in (("validation (ID)", config.data.val), ("test (OOD)", config.data.test)):
        acc, size, wrong = evaluate_split(model, tokenizer, split, device, max_new_tokens)
        print(f"{name}: exact_match={acc:.2%} ({size} samples)")
        for source, target, prediction in wrong:
            print(f"  {source} = {target} -> {prediction}")


if __name__ == "__main__":
    main()
