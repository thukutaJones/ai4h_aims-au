from transformers import LlamaConfig, LlamaForSequenceClassification

# we use eot in place for padding. In theory this is not correct, given the two tokens
# have different meaning. Still, we can do it here because, when eot is used in place
# of padding, this will be masked, and the model will actually never see the token (it's just
# a placeholder).
EOT = "<|eot_id|>"


def wrap_llama3_tokenizer(wrapped_tokenizer):
    eot_id = wrapped_tokenizer.convert_tokens_to_ids(EOT)
    wrapped_tokenizer.pad_token = EOT
    wrapped_tokenizer.pad_token_id = eot_id
    return wrapped_tokenizer


def wrap_llama3_model(pretrained_model_name_or_path, num_labels, wrapped_tokenizer):
    model = LlamaForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path, num_labels=num_labels
    )

    model.config.pad_token = wrapped_tokenizer.pad_token
    model.config.pad_token_id = wrapped_tokenizer.pad_token_id

    return model
