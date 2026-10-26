# English → German Transformer from Scratch (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N1glo1Hd9vntwiGoFHO9vfxAeIiMJMXr)

An encoder-decoder Transformer implemented from scratch in PyTorch and trained on Multi30k for English → German machine translation. Every component (multi-head self-attention, cross-attention, positional embeddings, padding and causal masks) is written by hand, without `nn.Transformer`.

## Results

| Version | Change | Translation BLEU (EN → DE, test set) |
|---|---|---|
| v1 | Baseline, Multi30k only | 12.5 |
| v2 | More training data | **21.0** |

| Auxiliary task | Score |
|---|---|
| Encoder masked-token reconstruction BLEU | 91.35 |

**What the two scores measure**
- **Translation BLEU (EN → DE):** how closely the generated German sentence matches the reference German sentence in the test set.
- **Encoder prediction BLEU:** how accurately the encoder reconstructs the original English sentence after some tokens are masked. It is a sanity check that the encoder learns useful representations, not a translation score, so it should not be compared with the translation BLEU.

## What I found

The v1 model (12.5 BLEU) produced mostly grammatical German, but it often swapped in the wrong word (see the table below). That pattern points to a **vocabulary and data-coverage problem**, not an architecture bug. Adding more training data lifted test BLEU from 12.5 to 21.0 without changing the model.

## Example translations (v1 model, 12.5 BLEU)

| English | Model output | Comment |
|---|---|---|
| Two dogs are playing in the park. | zwei hunde spielen im park. | Correct |
| A man is riding a bicycle. | ein mann fährt auf einem motorrad. | Wrong word: *motorrad* (motorcycle) instead of *fahrrad* |
| A woman is walking outside. | eine frau geht an einem strand. | Wrong word: *strand* (beach) |
| A young boy is playing with a ball. | ein kleiner junge spielt mit einem spielzeug. | Wrong word: *spielzeug* (toy) instead of *ball* |
| A group of people are sitting at a table. | eine gruppe von menschen sitzt auf einem tisch. | Wrong preposition: *auf* (on) instead of *an* (at) |

TODO: add 5 examples from the v2 model (21.0 BLEU) so readers can see the improvement.

## Architecture

- **Encoder:** token + positional embeddings, stacked blocks of multi-head self-attention and feed-forward layers, with residual connections and layer normalization.
- **Decoder:** masked (causal) self-attention, cross-attention over the encoder output, feed-forward layers.
- **Masks:** a padding mask so attention ignores `<pad>` tokens, and a causal mask so the decoder cannot see future target tokens.
- **Auxiliary encoder task:** masked-token reconstruction of the English source.

## Training setup

- **Dataset:** Multi30k (English–German image captions), plus additional parallel data in v2.
- TODO: number of training sentence pairs and the source of the extra data.
- TODO: BLEU library and settings (for example sacreBLEU, cased or lowercased).
- TODO: hyperparameters (layers, model dimension, heads, dropout, learning rate, epochs, batch size).

## How to run

1. Click the **Open in Colab** badge above.
2. Choose **Runtime → Run all**. The notebook downloads the data, trains the model, evaluates BLEU and prints example translations.
3. Use a GPU runtime (**Runtime → Change runtime type → T4 GPU**) for faster training.

## Limitations

- Multi30k is small (about 29k training pairs), so rare words are handled poorly.
- Word-level vocabulary: unseen words become `<unk>`.
- Greedy decoding only.
- Trained on short image-caption sentences, so it will not generalize to long or technical text.

## Next steps

- [ ] Subword tokenization (BPE or SentencePiece)
- [ ] Label smoothing and a learning-rate warmup schedule
- [ ] Beam search decoding
- [ ] Larger parallel corpus (for example IWSLT)
- [ ] Gradio demo on Hugging Face Spaces

## Repository structure

```
.
├── transformer_translation.ipynb   # model, training, evaluation
├── README.md
└── requirements.txt
```

## Author

**Priyansh Patel**: [GitHub](https://github.com/honeybadger-REX) · [LinkedIn](https://linkedin.com/in/priyansh-patel-659536275) · [Hugging Face](https://huggingface.co/priyanshbadge)
