import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", sql_output="polars")

with app.setup(hide_code=True):
    # Setup
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import gzip
    import psutil
    import threading

    alt.data_transformers.disable_max_rows()
    torch.set_num_threads(2)

    # Swap this one line to change the task vocabulary
    # CHARS = "🍕🍔🌮🍜🍣🍩🍰🍎🍋🍇"
    CHARS = "123456789"
    SEP = len(CHARS)

    PREDICT_DIGITS = 9
    VOCAB = len(CHARS) + 1  # +1 for <sep>
    SEQ_LEN = PREDICT_DIGITS * 2 + 1
    D_MODEL = 64
    N_HEADS = 2
    N_LAYERS = 2
    LR = 0.003
    BATCH = 64
    TARGET_ACC = 0.9
    CPU_TDP_W = 15.0

    NCA_STATES = len(CHARS)  # NCA cell alphabet matches token range
    NCA_H = 12
    NCA_W = 12
    NCA_T = 20
    NCA_WIN = SEQ_LEN - 1
    N_RULES = 20


@app.cell(hide_code=True)
def _():
    # general functions
    # It's just easier to store everything in one cell here. Feel free to split it up if you want.

    ## Power Tracker


    class PowerTracker:
        """
        Samples this process's CPU% every `interval` seconds in a background thread.
        Energy = integral of (cpu_fraction * TDP) dt  via trapezoidal rule.
        """

        def __init__(self, tdp_w=CPU_TDP_W, interval=0.05):
            self._proc = psutil.Process()
            self._tdp = tdp_w
            self._interval = interval
            self._samples = []  # [(timestamp, cpu_fraction), ...]
            self._stop_evt = threading.Event()
            self._t0 = None

        def start(self):
            self._proc.cpu_percent()  # seed call — first call always returns 0
            self._t0 = time.time()
            threading.Thread(target=self._loop, daemon=True).start()

        def _loop(self):
            while not self._stop_evt.wait(self._interval):
                frac = self._proc.cpu_percent() / 100.0
                self._samples.append((time.time(), frac))

        def stop(self):
            self._stop_evt.set()

        @property
        def energy_j(self):
            if len(self._samples) < 2:
                return 0.0
            j = 0.0
            for (t0, f0), (t1, f1) in zip(self._samples, self._samples[1:]):
                j += (f0 + f1) / 2 * self._tdp * (t1 - t0)
            return j

        @property
        def elapsed_s(self):
            return time.time() - self._t0


    ## Model


    char_to_id = {c: i for i, c in enumerate(CHARS)}
    id_to_char = {i: c for i, c in enumerate(CHARS)}


    class TinyGPT(nn.Module):
        def __init__(self, vocab=VOCAB, seq_len=SEQ_LEN):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab, D_MODEL)
            self.pos_emb = nn.Embedding(seq_len, D_MODEL)
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=D_MODEL,
                        nhead=N_HEADS,
                        dim_feedforward=2 * D_MODEL,
                        batch_first=True,
                        dropout=0.0,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(N_LAYERS)
                ]
            )
            self.ln_f = nn.LayerNorm(D_MODEL)
            self.head = nn.Linear(D_MODEL, vocab, bias=False)
            self.head.weight = self.tok_emb.weight  # weight tying

        def forward(self, x):
            _, T = x.shape
            mask = nn.Transformer.generate_square_subsequent_mask(T)
            h = self.tok_emb(x) + self.pos_emb(torch.arange(T))
            for blk in self.blocks:
                h = blk(h, src_mask=mask, is_causal=True)
            return self.head(self.ln_f(h))


    def predict_reverse(model, digits: str):
        tokens = [char_to_id[c] for c in digits if c in char_to_id]
        assert len(tokens) == PREDICT_DIGITS, (
            f"Expected {PREDICT_DIGITS} chars, got {len(tokens)}"
        )
        context = torch.tensor([tokens + [SEP]], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            for _ in range(PREDICT_DIGITS):
                logits = model(context)
                next_tok = logits[0, -1, :].argmax(-1, keepdim=True).unsqueeze(0)
                context = torch.cat([context, next_tok], dim=1)
        pred_ids = context[0, PREDICT_DIGITS + 1 :].tolist()
        exp_ids = list(reversed(tokens))
        return "".join(id_to_char.get(i, "?") for i in pred_ids), "".join(
            id_to_char.get(i, "?") for i in exp_ids
        )


    ## Dataset


    def make_batch(batch_size=BATCH):
        """
        Full sequence: [a b c d e f <sep> f e d c b a]
        Returns (x, y) each shape (B, 12) for causal next-token prediction.
        Accuracy is only measured on positions 7-12 (the reversed answer).
        """
        seqs = torch.randint(0, len(CHARS), (batch_size, PREDICT_DIGITS))
        sep = torch.full((batch_size, 1), SEP)
        full = torch.cat([seqs, sep, seqs.flip(1)], dim=1)  # (B, 13)
        return full[:, :-1], full[:, 1:]


    def inference(model, n=8):
        model.eval()
        seqs = torch.randint(0, 10, (n, 6))
        sep = torch.full((n, 1), SEP)
        # Feed only the input + sep; autoregressively predict the 6 answer tokens
        context = torch.cat([seqs, sep], dim=1)  # (n, 7)
        with torch.no_grad():
            for _ in range(PREDICT_DIGITS):
                logits = model(context)
                next_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # (n, 1)
                context = torch.cat([context, next_tok], dim=1)
        predicted = context[:, PREDICT_DIGITS + 1 :].tolist()

        print(f"  {'Input':<20} {'Expected':<20} {'Got':<20} OK?")
        print(f"  {'─' * 20} {'─' * 20} {'─' * 20} ────")
        n_correct = 0
        for i in range(n):
            inp = seqs[i].tolist()
            exp = list(reversed(inp))
            got = predicted[i]
            ok = "✓" if got == exp else "✗"
            if got == exp:
                n_correct += 1
            print(f"  {str(inp):<20} {str(exp):<20} {str(got):<20} {ok}")
        print(f"\n  {n_correct}/{n} sequences fully correct")


    ## Plot training logs


    def plot_training_log(logs):
        total_time_s = logs[-1]["time_s"]
        total_energy_j = logs[-1]["energy_j"]

        df = pd.DataFrame(logs)
        summary = f"Steps: {logs[-1]['step']} | Time: {total_time_s:.1f}s | Energy: {total_energy_j:.2f} J"

        base = alt.Chart(df).encode(x=alt.X("step:Q", title="Step"))

        loss_line = base.mark_line(color="steelblue", strokeWidth=2).encode(
            y=alt.Y(
                "loss:Q",
                title="Loss",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(titleColor="steelblue", labelColor="steelblue"),
            )
        )

        acc_line = base.mark_line(color="darkorange", strokeWidth=2).encode(
            y=alt.Y(
                "acc:Q",
                title="Accuracy",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(titleColor="darkorange", labelColor="darkorange"),
            )
        )

        chart = (
            alt.layer(loss_line, acc_line)
            .resolve_scale(y="independent")
            .properties(
                width=560,
                height=220,
                title=alt.TitleParams(
                    summary, anchor="middle", fontSize=12, color="gray"
                ),
            )
        )
        return chart


    ## Fine-tuning on reversal task


    def finetune(target_accuracy, model, label, power: PowerTracker):
        """Fine-tune on reversal task. Accepts an existing PowerTracker so that
        energy from a prior pre-training phase is included in the totals."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        log: list[dict] = []
        step = 0
        print(f"\n{label} — fine-tuning on reversal (target {TARGET_ACC:.0%})")

        while True:
            model.train()
            x, y = make_batch(BATCH)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 10 == 0:
                model.eval()
                with torch.no_grad():
                    xv, yv = make_batch(512)
                    acc = (
                        (
                            model(xv)[:, PREDICT_DIGITS:, :].argmax(-1)
                            == yv[:, PREDICT_DIGITS:]
                        )
                        .float()
                        .mean()
                        .item()
                    )

                e_j = power.energy_j  # cumulative since tracker was started
                log.append(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "acc": acc,
                        "time_s": power.elapsed_s,
                        "energy_j": e_j,
                    }
                )
                mo.output.replace(
                    mo.vstack(
                        [
                            mo.md(
                                f"**{label}** — step `{step}` | loss `{loss.item():.3f}` | acc `{acc:.1%}` | `{e_j:.2f}` J"
                            ),
                            plot_training_log(log),
                        ]
                    )
                )

                if acc >= target_accuracy:
                    power.stop()
                    break

            if step > 8000:
                power.stop()
                print("  Target not reached within 8000 steps")
                break

        return log


    ## NCA


    class NCARule(nn.Module):
        def __init__(self, n_states=NCA_STATES):
            super().__init__()
            self.n_states = n_states
            self.conv = nn.Conv2d(
                n_states, 16, 3, padding=1, padding_mode="circular"
            )
            self.mlp = nn.Sequential(
                nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, n_states)
            )
            with torch.no_grad():
                for layer in [self.conv, self.mlp[0], self.mlp[2]]:
                    nn.init.orthogonal_(layer.weight, gain=1.5)
                    nn.init.zeros_(layer.bias)

        def forward(self, x):
            h = self.conv(x).permute(0, 2, 3, 1)  # (B, H, W, 16)
            return self.mlp(h).permute(0, 3, 1, 2)  # (B, n_states, H, W)


    ## NCA dataset


    def sample_trajectory(rule):
        grid = torch.randint(0, NCA_STATES, (NCA_H, NCA_W))
        frames = [grid.clone()]
        with torch.no_grad():
            for _ in range(NCA_T - 1):
                x = (
                    F.one_hot(grid, NCA_STATES)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                grid = rule(x).squeeze(0).argmax(0)
                frames.append(grid.clone())
        return torch.stack(frames).reshape(-1)  # (NCA_T * H * W,)


    def make_nca_batch(nca_pool, batch_size=BATCH):
        """12-token windows of NCA cell values (0-9). Same vocab as reversal digits."""
        xs, ys = [], []
        for _ in range(batch_size):
            flat = nca_pool[torch.randint(N_RULES, (1,)).item()]
            start = torch.randint(0, flat.shape[0] - NCA_WIN - 1, (1,)).item()
            seq = flat[start : start + NCA_WIN + 1]
            xs.append(seq[:-1])
            ys.append(seq[1:])
        return torch.stack(xs), torch.stack(ys)

    return (
        NCARule,
        PowerTracker,
        TinyGPT,
        finetune,
        make_nca_batch,
        plot_training_log,
        predict_reverse,
        sample_trajectory,
    )


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Can synthetic structure help a model learn faster?

    We test a simple hypothesis from the paper
    [*Training Language Models via Neural Cellular Automata*](https://arxiv.org/abs/2603.10055):

    > Pre-training a transformer on **structured synthetic sequences** before it ever sees real task data
    > lets it reach the same accuracy in **fewer steps** and with **less energy**.

    We use a toy task, **reversing a {PREDICT_DIGITS} character sequence**mall enough to run on a CPU in seconds.

    | Run | What happens |
    |-----|-------------|
    | **Scratch** | Train from random weights directly on the reversal task |
    | **NCA** | Pre-train on Neural Cellular Automata sequences first, then fine-tune on reversal |

    We record steps and real CPU energy (via `psutil`) for both runs and compare at the end.
    """)
    return


@app.cell(hide_code=True)
def _(TinyGPT):
    mo.vstack([
        mo.md("## The Model: TinyGPT"),
        mo.callout(
            mo.md("**Both runs use the exact same model.** The only difference is *what it reads before the test* — one starts cold, one has already seen NCA sequences."),
            kind="info"
        ),
        mo.md(f"""
    Think of it as **next-token autocomplete** — like your phone's keyboard suggestions.

    It reads tokens left-to-right and predicts what comes next at each position.
    A *causal mask* prevents it from peeking ahead — it can only use what it has already seen.

    Two learned lookup tables feed into it:
    - **Token embeddings** — a vector for each symbol in the vocabulary
    - **Position embeddings** — a vector for each slot in the sequence

    These are summed and passed through {N_HEADS} transformer blocks, then projected back to a score over every possible next
    token.
    """),
        mo.accordion({
        "Technical details": mo.md(f"""
    Input tokens are embedded as $\mathbf{{h}}_0 = \mathbf{{E}}[x] + \mathbf{{P}}[\text{{pos}}]$,
    where $\mathbf{{E}} \in \mathbb{{R}}^{{V \\times d}}$ and $\mathbf{{P}} \in \mathbb{{R}}^{{T_{{\\max}} \\times d}}$.

    Each of the $L$ **Pre-LayerNorm** blocks applies:
    1. Causal multi-head self-attention ($H$ heads)
    2. Position-wise FFN: $d \\to 2d \\to d$, GELU activation, no dropout

    A final layer norm feeds a **weight-tied** linear head ($\mathbf{{W}}_{{\\text{{head}}}} = \mathbf{{E}}$).
    Trained with teacher-forced cross-entropy.

    | | |
    |---|---|
    | Model dim $d$ | {D_MODEL} |
    | Layers $L$ | {N_LAYERS} |
    | Heads $H$ | {N_HEADS} |
    | Vocab $V$ | {VOCAB} |
    | Max seq len | {SEQ_LEN} |
    | Parameters | ~{f"{sum(p.numel() for p in TinyGPT().parameters() if p.requires_grad) / (2**10):.0f}"} K |
    """),
        }),
    ])
    return


@app.cell(hide_code=True)
def _():
    import random as _random

    random_chars = _random.choices(list(CHARS), k=PREDICT_DIGITS)
    sample_input = "".join(random_chars)

    mo.md(f"""
    ## The task: sequence reversal

    The model receives {PREDICT_DIGITS} tokens + a separator `<sep>`, then must output them in reverse order:

    ```
    Input:   {" ".join(random_chars)}  <sep>  {" ".join(["?"] * PREDICT_DIGITS)}
    Target:                            {" ".join(reversed(random_chars))}
    ```

    Accuracy is measured only on the {PREDICT_DIGITS} answer positions.

    **Before any training**, the model outputs random noise. Try it below:
    """)
    return (sample_input,)


@app.cell
def _(sample_input):
    scratch_input = mo.ui.text(
        label="Enter text:",
        placeholder=sample_input,
        value=sample_input,
    )
    scratch_input
    return (scratch_input,)


@app.cell
def _(TinyGPT, predict_reverse, scratch_input):
    input_string = scratch_input.value.strip()
    predicted, expected = predict_reverse(TinyGPT(), input_string)
    mo.callout(
        mo.md(
            f"Untrained model (random weights) — input `{input_string}` → expected **{expected}**, got **{predicted}**"
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run 1: Train from scratch

    We train TinyGPT directly on the reversal task from random weights. CPU usage is sampled continuously and integrated over time to give a real energy estimate in joules.
    """)
    return


@app.cell
def _():
    # target_acc_input = mo.ui.number(start=0.1, stop=0.99, step=0.02, label="Target accuracy for fine-tuning:", value=TARGET_ACC)
    target_acc_input = mo.ui.slider(start=0.1, stop=0.99, step=0.02, label="Target accuracy for fine-tuning:", value=TARGET_ACC)
    train_scratch_button = mo.ui.run_button(label="Train TinyGPT")
    mo.vstack([
        "Set your target accuracy below, then train the model. We'll stop as soon as it hits that threshold and record everything.",
        target_acc_input,
        train_scratch_button
    ])
    return target_acc_input, train_scratch_button


@app.cell
def _(PowerTracker, TinyGPT, finetune, target_acc_input, train_scratch_button):
    mo.stop(not train_scratch_button.value)
    # run scratch model
    scratch_model = TinyGPT()
    scratch_power = PowerTracker()
    scratch_power.start()
    scratch_log = finetune(target_acc_input.value, scratch_model, "Scratch", scratch_power)
    return scratch_log, scratch_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Training complete. The same input that stumped it moments ago should now produce the right answer.
    """)
    return


@app.cell
def _(sample_input):
    trained_scratch_input = mo.ui.text(
        label="Test trained scratch model:",
        placeholder=sample_input,
        value=sample_input,
    )
    trained_scratch_input
    return (trained_scratch_input,)


@app.cell
def _(
    plot_training_log,
    predict_reverse,
    scratch_log,
    scratch_model,
    trained_scratch_input,
):
    trained_input_string = trained_scratch_input.value.strip()
    trained_predicted, trained_expected = predict_reverse(
        scratch_model, trained_input_string
    )
    _correct = trained_predicted == trained_expected


    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Trained scratch model — input `{trained_input_string}` → expected **{trained_expected}**, got **{trained_predicted}**"
                ),
                kind="success" if _correct else "danger",
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{scratch_log[-1]['step']}", label="Steps to target"
                    ),
                    mo.stat(
                        value=f"{scratch_log[-1]['energy_j']:.2f} J",
                        label="Energy used",
                    ),
                    mo.stat(
                        value=f"{scratch_log[-1]['time_s']:.1f} s",
                        label="Wall-clock time",
                    ),
                    mo.stat(
                        value=f"{scratch_log[-1]['acc']:.1%}",
                        label="Final accuracy",
                    ),
                ]
            ),
            mo.accordion(
                {"Scratch training charts": plot_training_log(scratch_log)}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run 2: NCA pre-pre-training

    As mentioned earlier, the core idea of this research is that pre-training a transformer on Neural Cellular Automata (NCA) sequences provides it with "structural intuitions" (ie fundamental patterns and logic), that it can later reuse to learn natural language much faster.

    This is very interesting because it could reduce the need for large-scale pre-training on natural data, which is costly and environmentally impactful. If synthetic structure can bootstrap learning, we might train more efficient models with less real data and energy.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We expose the model to NCA sequences first, structured, non-linguistic patterns designed to build pattern-recognition before any task-specific training. Use the slider to choose how many pre-training steps to run.
    """)
    return


@app.cell
def _(NCARule, sample_trajectory):
    nca_pool = [sample_trajectory(NCARule()) for _ in range(N_RULES)]
    return (nca_pool,)


@app.cell
def _():
    nca_pre_training_slider = mo.ui.slider(label="NCA pre-training steps", start=0, stop=32*8, step=32, value=128, show_value=True)
    nca_pre_train_button = mo.ui.run_button(label="Run NCA pre-training")
    mo.vstack([
        nca_pre_training_slider,
        nca_pre_train_button
    ])
    return nca_pre_train_button, nca_pre_training_slider


@app.cell(hide_code=True)
def _(
    PowerTracker,
    TinyGPT,
    make_nca_batch,
    nca_pool,
    nca_pre_train_button,
    nca_pre_training_slider,
    plot_training_log,
):
    # pre-pretraining

    mo.stop(not nca_pre_train_button.value)

    NCA_PRETRAIN = nca_pre_training_slider.value

    nca_model = TinyGPT()

    # Start tracker before pre-training so total energy covers both phases
    nca_power = PowerTracker()
    nca_power.start()

    print(f"\nNCA pre-training ({NCA_PRETRAIN} steps on {N_RULES} NCA rules)")
    pre_opt = torch.optim.AdamW(nca_model.parameters(), lr=LR)
    pre_log: list[dict] = []

    for step in range(1, NCA_PRETRAIN + 1):
        nca_model.train()
        xn, yn = make_nca_batch(nca_pool, BATCH)
        loss = F.cross_entropy(nca_model(xn).reshape(-1, VOCAB), yn.reshape(-1))
        pre_opt.zero_grad()
        loss.backward()
        pre_opt.step()
        if step % (NCA_PRETRAIN // 10) == 0:
            nca_model.eval()
            with torch.no_grad():
                xv, yv = make_nca_batch(nca_pool, 256)
                acc = (nca_model(xv).argmax(-1) == yv).float().mean().item()
            e_j = nca_power.energy_j
            pre_log.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "acc": acc,
                    "energy_j": e_j,
                    "time_s": nca_power.elapsed_s,
                }
            )
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md(
                            f"**NCA pre-training** — step `{step}/{NCA_PRETRAIN}` | loss `{loss.item():.3f}` | acc `{acc:.1%}` | `{e_j:.2f}` J"
                        ),
                        plot_training_log(pre_log),
                    ]
                )
            )

    pre_energy_j = nca_power.energy_j  # snapshot at end of pre-training
    return nca_model, nca_power, pre_energy_j, pre_log


@app.cell
def _():
    mo.md(r"""
    ### Stage 1 done: now test reversal (should still be wrong)

    The model has learned NCA dynamics but has **never seen the reversal task**.
    Test it below and expect wrong answers, that is the point:
    """)
    return


@app.cell
def _(sample_input):
    nca_trained_input = mo.ui.text(
        label="Test NCA model before fine-tuning:",
        placeholder=sample_input,
        value=sample_input,
    )
    nca_trained_input
    return (nca_trained_input,)


@app.cell(hide_code=True)
def _(nca_model, nca_trained_input, predict_reverse):
    nca_pretrained_pred, nca_pretrained_exp = predict_reverse(
        nca_model, nca_trained_input.value.strip()
    )
    mo.callout(
        mo.md(
            f"Pre-trained (no fine-tuning yet) — expected **{nca_pretrained_exp}**, got **{nca_pretrained_pred}** *(wrong is expected here)*"
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stage 2: fine-tune on the reversal task

    Same training loop as the scratch model.

    The power tracker continues from Stage 1, so we already includes pre-training cost.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_nca_button = mo.ui.run_button(label="Fine-tune NCA model")
    train_nca_button
    return (train_nca_button,)


@app.cell
def _(finetune, nca_model, nca_power, target_acc_input, train_nca_button):
    mo.stop(not train_nca_button.value)
    nca_log = finetune(target_acc_input.value, nca_model, "NCA", nca_power)
    return (nca_log,)


@app.cell
def _(nca_log, plot_training_log, scratch_log):
    _nca_steps = nca_log[-1]["step"]
    _nca_energy = nca_log[-1]["energy_j"]
    _scratch_steps = scratch_log[-1]["step"]
    _scratch_energy = scratch_log[-1]["energy_j"]
    _speedup = _scratch_steps / _nca_steps
    _saving = (1 - _nca_energy / _scratch_energy) * 100

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_scratch_steps}", label="Scratch fine-tune steps"
                    ),
                    mo.stat(value=f"{_nca_steps}", label="NCA fine-tune steps"),
                    mo.stat(value=f"{_speedup:.1f}×", label="Step speedup"),
                ]
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_scratch_energy:.2f} J",
                        label="Scratch total energy",
                    ),
                    mo.stat(
                        value=f"{_nca_energy:.2f} J",
                        label="NCA total energy (incl. pre-train)",
                    ),
                    mo.stat(value=f"{_saving:.1f}%", label="Energy saved"),
                ]
            ),
            mo.callout(
                mo.md(
                    f"The NCA model reached **{nca_log[-1]['acc']:.1%}** accuracy in **{_nca_steps} fine-tuning steps** "
                    f"vs **{_scratch_steps} steps** for scratch — a **{_speedup:.1f}× speedup**. "
                    f"\n\nEven counting pre-training energy, total cost dropped by **{_saving:.1f}%**."
                ),
                kind="success" if _saving > 0 else "info",
            ),
            mo.accordion({"NCA training charts": plot_training_log(nca_log)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Results
    """)
    return


@app.cell
def _(nca_log, pre_energy_j, scratch_log):
    s_steps = scratch_log[-1]["step"]
    n_steps = nca_log[-1]["step"]
    s_e     = scratch_log[-1]["energy_j"]
    n_e     = nca_log[-1]["energy_j"]

    print("\n" + "─" * 55)
    print(f"  {'':30s} {'Scratch':>10} {'NCA':>10}")
    print(f"  {'Fine-tune steps to target':30s} {s_steps:>10} {n_steps:>10}")
    print(f"  {'Pre-train energy (J)':30s} {'—':>10} {pre_energy_j:>10.2f}")
    print(f"  {'Fine-tune energy (J)':30s} {s_e:>10.2f} {n_e - pre_energy_j:>10.2f}")
    print(f"  {'Total energy (J)':30s} {s_e:>10.2f} {n_e:>10.2f}")
    print(f"  {'Speedup (fine-tune steps)':30s} {'':>10} {s_steps/n_steps:>9.1f}×")
    print(f"  {'Total energy saving':30s} {'':>10} {(1 - n_e/s_e)*100:>9.1f}%")
    print("─" * 55)
    return


@app.cell
def _(nca_log, pre_log: list[dict], scratch_log):
    scratch_df = pd.DataFrame([{**row, "model": "Scratch"} for row in scratch_log])
    nca_ft_df = pd.DataFrame([{**row, "model": "NCA"} for row in nca_log])
    compare_df = pd.concat([scratch_df, nca_ft_df], ignore_index=True)

    color = alt.Color(
        "model:N",
        scale=alt.Scale(
            domain=["Scratch", "NCA"], range=["steelblue", "darkorange"]
        ),
    )

    acc_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="Fine-tune step"),
            y=alt.Y(
                "acc:Q", title="Reversal accuracy", scale=alt.Scale(domain=[0, 1])
            ),
            color=color,
        )
        .properties(title="Accuracy vs fine-tune steps", width=350, height=220)
    )

    energy_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("energy_j:Q", title="Fine-tune energy (J)"),
            y=alt.Y(
                "acc:Q", title="Reversal accuracy", scale=alt.Scale(domain=[0, 1])
            ),
            color=color,
        )
        .properties(title="Accuracy vs fine-tune energy", width=350, height=220)
    )

    pre_df = pd.DataFrame(pre_log)
    pre_chart = (
        alt.Chart(pre_df)
        .mark_line(color="purple")
        .encode(
            x=alt.X("step:Q", title="Pre-train step"),
            y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(zero=False)),
        )
        .properties(title="NCA pre-train loss", width=250, height=220)
    )

    chart = pre_chart | acc_compare | energy_compare
    chart
    return


@app.cell
def _():
    mo.md(r"""
    ---
    *Experiment based on [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055) — built with [marimo](https://marimo.io)*
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
