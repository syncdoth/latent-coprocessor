import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire
import wandb


class CoProcessor(torch.nn.Module):
    def __init__(self, model_name_or_path: str, n_latents: int):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.latents = torch.nn.Parameter(
            torch.randn(n_latents, self.model.config.hidden_size)
        )

    def forward(self, batch_size: int, past_key_values):
        latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)
        outputs = self.model(
            past_key_values=past_key_values,
            inputs_embeds=latents,
            output_hidden_states=True,
            use_cache=True,
        )
        return outputs.hidden_states[-1], outputs.past_key_values


def train(
    model,
    coprocessor,
    optimizer,
    dataloader,
    wandb_run=None,
    max_epochs: int = 1,
    device: torch.device | None = None,
):
    for epoch in range(max_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    use_cache=True,
                )

            latents, past_key_values = coprocessor(
                batch_size=batch.shape[0], past_key_values=outputs.past_key_values
            )

            outputs = model(
                input_ids=batch["labels"].to(device),
                past_key_values=past_key_values,
                use_cache=True,
            )

            loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1].reshape(-1, outputs.logits.shape[-1]),
                batch["labels"][:, 1:].reshape(-1),
            )

            loss.backward()
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item()})


def main(
    model_name_or_path: str,
    n_latents: int,
    dataset_name: str,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_epochs: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    coprocessor = CoProcessor(model, n_latents).to(device)
    optimizer = torch.optim.AdamW(
        coprocessor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    dataset = load_dataset(dataset_name)
    dataset = dataset.map(
        lambda x: {k: v[0] for k, v in tokenizer(x["text"], return_tensors="pt")}
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    wandb_run = wandb.init(
        project="latent-model", name=f"latents-{n_latents}-{dataset_name}"
    )
    train(model, coprocessor, optimizer, dataloader, wandb_run, max_epochs)


if __name__ == "__main__":
    fire.Fire(main)
