import torch as th

import numpy as np
from tqdm import tqdm


def train(
    train_data_loader,
    probe,
    bert,
    loss_fct,
    optimizer,
    dev_data_loader=None,
    scheduler=None,
):
    # Train the probe
    probe.train()
    train_loss, dev_loss = 0, 0
    train_acc, dev_acc = 0, 0
    for batch in tqdm(train_data_loader, desc="[Train]"):
        optimizer.zero_grad()
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(probe.device),
            text_token_type_ids.to(probe.device),
            text_attention_mask.to(probe.device),
            label.to(probe.device),
        )

        with th.no_grad():
            outputs = bert(
                text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs[2]
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )

        logits = probe(sequence_output)
        l = loss_fct(logits.view(-1, 2), label.view(-1))
        train_loss += l.item()
        l.backward()
        optimizer.step()

        train_acc += (logits.argmax(-1) == label).sum().item()
    train_loss = train_loss / len(train_data_loader.dataset)
    train_acc = train_acc / len(train_data_loader.dataset)

    if dev_data_loader is not None:
        probe.eval()
        for batch in tqdm(dev_data_loader, desc="[Dev]"):
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                text_input_ids.to(probe.device),
                text_token_type_ids.to(probe.device),
                text_attention_mask.to(probe.device),
                label.to(probe.device),
            )
            with th.no_grad():
                outputs = bert(
                    text_input_ids,
                    attention_mask=text_attention_mask,
                    token_type_ids=text_token_type_ids,
                    output_hidden_states=True,
                )
                hidden_states = outputs[2]
                sequence_output = (
                    hidden_states[probe.layer_num]
                    .to(probe.device)
                    .to(probe.default_dtype)
                )
                logits = probe(sequence_output)

                l = loss_fct(logits.view(-1, 2), label.view(-1))
                dev_loss += l.item()

            dev_acc += (logits.argmax(-1) == label).sum().item()
        # Adjust the learning rate
        if scheduler is not None:
            scheduler.step(dev_loss)

        dev_loss = dev_loss / len(dev_data_loader.dataset)
        dev_acc = dev_acc / len(dev_data_loader.dataset)

    return (
        train_loss,
        train_acc,
        dev_loss,
        dev_acc,
    )
