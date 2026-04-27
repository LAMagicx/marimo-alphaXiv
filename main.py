import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium", sql_output="polars")


@app.cell(hide_code=True)
def _():
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
    CHARS = "0123456789"
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
    return (
        BATCH,
        CHARS,
        CPU_TDP_W,
        D_MODEL,
        F,
        LR,
        NCA_H,
        NCA_STATES,
        NCA_T,
        NCA_W,
        NCA_WIN,
        N_HEADS,
        N_LAYERS,
        N_RULES,
        PREDICT_DIGITS,
        SEP,
        SEQ_LEN,
        TARGET_ACC,
        VOCAB,
        alt,
        gzip,
        mo,
        nn,
        np,
        pd,
        psutil,
        threading,
        time,
        torch,
    )


@app.cell(hide_code=True)
def helper_functions(
    BATCH,
    CHARS,
    CPU_TDP_W,
    D_MODEL,
    F,
    LR,
    NCA_H,
    NCA_STATES,
    NCA_T,
    NCA_W,
    NCA_WIN,
    N_HEADS,
    N_LAYERS,
    N_RULES,
    PREDICT_DIGITS,
    SEP,
    SEQ_LEN,
    TARGET_ACC,
    VOCAB,
    alt,
    mo,
    nn,
    pd,
    psutil,
    threading,
    time,
    torch,
):
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
            self._proc.cpu_percent()  # seed call  -  first call always returns 0
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
        print(f"\n{label}: fine-tuning on reversal (target {TARGET_ACC:.0%})")

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
                                f"**{label}** step `{step}` | loss `{loss.item():.3f}` | acc `{acc:.1%}` | `{e_j:.2f}` J"
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
        def __init__(self, n_states=NCA_STATES, seed=None):
            super().__init__()
            self.n_states = n_states
            self.seed = seed
            if seed is not None:
                torch.manual_seed(seed)
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
        make_batch,
        make_nca_batch,
        plot_training_log,
        predict_reverse,
        sample_trajectory,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""
    ## Introduction to this Notebook

    Training Large Language Models is not easy. It's expensive in both time and energy. What if there was a way to decrease both whilst improving accuracy and that it's based on a techinc that was first thought about in 1970.

    Welcome to Training Language Models via Neural Cellular Automata.

    Traditional approaches for training Large Language Models (LLMs) use two training steps. A pre-training step where it's trained by reading enormous amounts of text and learning to predict what comes next then a fine-tuning step where the model is trained to solve a specific problem by using questions and answers.

    Think about how children learn to read. You don't hand a five-year-old The Lord of the Rings on day one. You start with picture books, lots of bold colours, simple shapes, repeating patterns. The pictures and patterns build up the child's cognitive machinery before the complexity of real language arrives. Then, little by little, the images fade out and the words take over.

    The paper proposes a similar approach, they define a Pre-pre-training step that works on a similar idea: before exposing the model to the full complexity of human language, give it something simpler but deeply structured to warm up on.
    But here's where it gets surprising. The "picture books" used in this paper contain no language at all; no words, no sentences, no meaning. Just abstract visual patterns that follow hidden rules. And yet, warming up on these patterns makes the model significantly better at language afterwards.

    ![llm-training-steps](public/llm_training_steps.png)

    I learn best by example, so let's look at small demonstration that captures the core idea: does a model that warms up on structured patterns learn faster than one starting from scratch?
    """)
    return


@app.cell(hide_code=True)
def part1_demo_header(D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, TinyGPT, VOCAB, mo):
    mo.vstack(
        [
            mo.md(r"""
    # Part 1 Demonstration

    We can't train a LLM locally on a CPU, but we can capture the core breakthrough of the NCA paper with a controlled experiment. 

    We'll compare two training regimes for a **TinyGPT** model:
    1. **Run 1 (Scratch):** The model starts with random weights and is trained directly on a "reversal" task (learning to output a sequence in reverse).
    2. **Run 2 (NCA):** The model first undergoes **Pre-pretraining** on synthetic NCA patterns (abstract visual structures with no linguistic meaning) before being fine-tuned on the same reversal task.

    ![experiment-flow](public/experiment_flow.png)

    ### The Hypothesis
    Even including the energy cost of the NCA pre-training phase, the total training cost (energy + time) should be lower than starting from scratch. We are looking for a **speedup in fine-tuning** that more than offsets the initial investment in synthetic structure.
    """),
            mo.accordion(
                {
                    "Technical details": mo.md(f"""
    Input tokens are embedded as $\\mathbf{{h}}_0 = \\mathbf{{E}}[x] + \\mathbf{{P}}[\\text{{pos}}]$,
    where $\\mathbf{{E}} \\in \\mathbb{{R}}^{{V \\times d}}$ and $\\mathbf{{P}} \\in \\mathbb{{R}}^{{T_{{\\max}} \\times d}}$.

    Each of the $L$ **Pre-LayerNorm** blocks applies:
    1. Causal multi-head self-attention ($H$ heads)
    2. Position-wise FFN: $d \\to 2d \\to d$, GELU activation, no dropout

    A final layer norm feeds a **weight-tied** linear head ($\\mathbf{{W}}_{{\\text{{head}}}} = \\mathbf{{E}}$).
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
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def sample_input_generation(CHARS, PREDICT_DIGITS, mo):
    import random as _random

    random_chars = _random.choices(list(CHARS), k=PREDICT_DIGITS)
    sample_input = "".join(random_chars)

    scratch_input_slider = mo.ui.slider(
        label="Choose a number: ",
        start=0,
        stop=int("9" * PREDICT_DIGITS),
        value=int(sample_input),
        show_value=True,
    )

    mo.vstack(
        [
            mo.md(f"""
    ## Run 1: Training from Scratch

    Before we introduce any synthetic structure, we need a baseline. We'll train TinyGPT directly on the **Reversal Task**: given {PREDICT_DIGITS} digits and a separator, the model must output them in reverse order.

    ```
    Input:   {" ".join(random_chars)}  <sep>  {" ".join(["?"] * PREDICT_DIGITS)}
    Target:                            {" ".join(reversed(random_chars))}
    ```

    Without any training, the model's weights are random. It has no concept of position, digits, or "reversing" and just outputs high-entropy noise.
    """),
            scratch_input_slider,
        ]
    )
    return sample_input, scratch_input_slider


@app.cell(hide_code=True)
def scratch_model_prediction(
    TinyGPT,
    mo,
    predict_reverse,
    scratch_input_slider,
):
    input_string = str(scratch_input_slider.value)
    predicted, expected = predict_reverse(TinyGPT(), input_string)

    mo.vstack(
        [
            # mo.md(f"### Live Test: Untrained Model"),
            mo.md(f"**Current Input:** `{input_string}`"),
            mo.callout(
                mo.md(
                    f"The model predicted **{predicted}** (expected **{expected}**). "
                    f"\n\nAs expected, the random weights produce a nonsense sequence."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def baseline_training_md(mo):
    mo.md(r"""
    ## Baseline Training

    We now train the scratch model. We monitor CPU usage continuously to estimate the energy cost of reaching our target accuracy.
    """)
    return


@app.cell(hide_code=True)
def target_accuracy_controls(TARGET_ACC, mo):
    target_acc_input = mo.ui.slider(
        start=0.5, stop=0.99, step=0.01, label="Target accuracy:", value=TARGET_ACC, show_value=True
    )
    train_scratch_button = mo.ui.run_button(
        label="Start Scratch Training", kind="neutral"
    )

    mo.vstack(
        [
            mo.md(
                "Set the threshold for 'success'. The model will train until it consistently reaches this accuracy on the reversal task."
            ),
            mo.hstack(
                [target_acc_input, train_scratch_button], align="center", gap="2rem", justify="start"
            ),
        ]
    )
    return target_acc_input, train_scratch_button


@app.cell(hide_code=True)
def run_scratch_training(
    PowerTracker,
    TinyGPT,
    finetune,
    mo,
    target_acc_input,
    train_scratch_button,
):
    mo.stop(not train_scratch_button.value)
    # run scratch model
    scratch_model = TinyGPT()
    scratch_power = PowerTracker()
    scratch_power.start()
    scratch_log = finetune(target_acc_input.value, scratch_model, "Scratch", scratch_power)
    return scratch_log, scratch_model


@app.cell(hide_code=True)
def scratch_results_md(mo):
    mo.md("""
    ### Training Results

    Now that the weights have been optimized for the reversal task, let's verify the performance and check the efficiency metrics.
    """)
    return


@app.cell
def scratch_test_slider(PREDICT_DIGITS, mo, sample_input):
    trained_scratch_input_slider = mo.ui.slider(
        label="Test the trained model: ",
        start=0,
        stop=int("9" * PREDICT_DIGITS),
        value=int(sample_input),
        show_value=True,
    )
    trained_scratch_input_slider
    return (trained_scratch_input_slider,)


@app.cell(hide_code=True)
def scratch_performance_metrics(
    mo,
    predict_reverse,
    scratch_log,
    scratch_model,
    trained_scratch_input_slider,
):
    trained_input_string = str(trained_scratch_input_slider.value)
    trained_predicted, trained_expected = predict_reverse(
        scratch_model, trained_input_string
    )
    _correct = trained_predicted == trained_expected

    mo.vstack(
        [
            mo.md(
                f"**Input:** `{trained_input_string}` → **Output:** `{trained_predicted}`"
            ),
            mo.callout(
                mo.md(
                    f"The model successfully reversed the input!"
                    if _correct
                    else "The model is still struggling with this specific input."
                ),
                kind="success" if _correct else "danger",
            ),
            mo.md("#### Performance Statistics"),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{scratch_log[-1]['step']}", label="Steps Taken"
                    ),
                    mo.stat(
                        value=f"{scratch_log[-1]['energy_j']:.2f} J",
                        label="Energy Consumed",
                    ),
                    mo.stat(
                        value=f"{scratch_log[-1]['time_s']:.1f} s",
                        label="Wall-clock Time",
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def nca_pretraining_intro_md(mo):
    mo.md(r"""
    ## Run 2: NCA Pre-pretraining

    Now we test the paper's core hypothesis: **Does warming up on abstract, structured patterns make the model a better learner?**

    We'll expose TinyGPT to NCA sequences of non-linguistic, synthetic data, building "structural intuitions" before it ever sees the reversal task. This is the **Pre-pretraining** phase.
    """)
    return


@app.cell(hide_code=True)
def nca_warmup_md(mo):
    mo.md(r"""
    ### Stage 1: Synthetic Warmup
    The model predicts the next state in evolving NCA grids. These patterns are designed to be structured but abstract. Use the slider to choose the duration of this "picture book" phase.
    """)
    return


@app.cell(hide_code=True)
def initialize_nca_pool(NCARule, N_RULES, sample_trajectory):
    nca_pool = [sample_trajectory(NCARule()) for _ in range(N_RULES)]
    return (nca_pool,)


@app.cell(hide_code=True)
def nca_pretraining_controls(mo):
    nca_pre_training_slider = mo.ui.slider(
        label="NCA pre-training steps",
        start=32,
        stop=256,
        step=16,
        value=96,
        show_value=True,
    )
    nca_pre_train_button = mo.ui.run_button(label="Start NCA Pre-pretraining")

    mo.vstack(
        [
            mo.md("Choose how many steps to spend on synthetic patterns:"),
            mo.hstack(
                [nca_pre_training_slider, nca_pre_train_button],
                align="center",
                gap="2rem",
                justify="start"
            ),
        ]
    )
    return nca_pre_train_button, nca_pre_training_slider


@app.cell(hide_code=True)
def run_nca_pretraining(
    BATCH,
    F,
    LR,
    N_RULES,
    PowerTracker,
    TinyGPT,
    VOCAB,
    make_nca_batch,
    mo,
    nca_pool,
    nca_pre_train_button,
    nca_pre_training_slider,
    plot_training_log,
    torch,
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
                            f"**NCA pre-training** step `{step}/{NCA_PRETRAIN}` | loss `{loss.item():.3f}` | acc `{acc:.1%}` | `{e_j:.2f}` J"
                        ),
                        plot_training_log(pre_log),
                    ]
                )
            )

    pre_energy_j = nca_power.energy_j  # snapshot at end of pre-training
    return nca_model, nca_power, pre_log


@app.cell(hide_code=True)
def checkpoint_zero_shot_md(mo):
    mo.md(r"""
    ### Checkpoint: Zero-Shot Performance
    At this point, the model has learned the "logic" of NCA sequences, but it has **never seen a digit reversal**.

    If we test it now, it should still fail the reversal task because it hasn't been fine-tuned yet. The goal here isn't task-knowledge, but **structural readiness**.
    """)
    return


@app.cell(hide_code=True)
def nca_zero_shot_slider(PREDICT_DIGITS, mo, sample_input):
    nca_trained_input_slider = mo.ui.slider(
        label="Test NCA model before fine-tuning:",
        start=int("".join([str(_x) for _x in "1" * PREDICT_DIGITS])),
        stop=int("".join([str(_x) for _x in "9" * PREDICT_DIGITS])),
        value=int(sample_input),
        show_value=True,
    )
    nca_trained_input_slider
    return (nca_trained_input_slider,)


@app.cell(hide_code=True)
def nca_zero_shot_results(
    mo,
    nca_model,
    nca_trained_input_slider,
    predict_reverse,
):
    nca_trained_input_string = str(nca_trained_input_slider.value)
    nca_pretrained_pred, nca_pretrained_exp = predict_reverse(
        nca_model, nca_trained_input_string
    )

    mo.vstack(
        [
            mo.md(
                f"**Input:** `{nca_trained_input_string}` → **Output:** `{nca_pretrained_pred}`"
            ),
            mo.callout(
                mo.md(
                    f"As expected, the model fails (it hasn't been fine-tuned for reversal yet). "
                    f"\n\nNext, we'll see how quickly it can learn the task compared to the scratch model."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def nca_finetune_intro_md(mo):
    mo.md(r"""
    ### Stage 2: Fine-tuning on Reversal
    Now we fine-tune the NCA-warmed model on the reversal task using the same target accuracy as before.

    The energy tracker has been running since Stage 1, so the final results will include the **total cost** of both phases.
    """)
    return


@app.cell(hide_code=True)
def nca_finetune_button(mo):
    train_nca_button = mo.ui.run_button(
        label="Start NCA Fine-tuning", kind="success"
    )
    train_nca_button
    return (train_nca_button,)


@app.cell(hide_code=True)
def run_nca_finetuning(
    finetune,
    mo,
    nca_model,
    nca_power,
    target_acc_input,
    train_nca_button,
):
    mo.stop(not train_nca_button.value)
    nca_log = finetune(target_acc_input.value, nca_model, "NCA", nca_power)
    return (nca_log,)


@app.cell(hide_code=True)
def nca_post_ft_slider(PREDICT_DIGITS, mo, sample_input):
    nca_post_ft_slider = mo.ui.slider(
        label="Test NCA model after fine-tuning:",
        start=int("".join([str(_x) for _x in "1" * PREDICT_DIGITS])),
        stop=int("".join([str(_x) for _x in "9" * PREDICT_DIGITS])),
        value=int(sample_input),
        show_value=True,
    )
    nca_post_ft_slider
    return (nca_post_ft_slider,)


@app.cell(hide_code=True)
def nca_post_ft_results(mo, nca_model, nca_post_ft_slider, predict_reverse):
    nca_post_input_string = str(nca_post_ft_slider.value)
    nca_post_pred, nca_post_exp = predict_reverse(nca_model, nca_post_input_string)
    _nca_correct = nca_post_pred == nca_post_exp

    mo.vstack(
        [
            mo.md(
                f"**Input:** `{nca_post_input_string}` → **Output:** `{nca_post_pred}`"
            ),
            mo.callout(
                mo.md(
                    f"The NCA-pre-trained model successfully reversed the input!"
                    if _nca_correct
                    else "The model missed this one, but it learned significantly faster than the baseline."
                ),
                kind="success" if _nca_correct else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def final_results_md(mo):
    mo.md(r"""
    ## Final Results: The Efficiency Verdict

    The primary measure of success for NCA pre-pretraining is whether the total energy spent (Pre-training + Fine-tuning) is less than the energy spent training from scratch or wheather the total time spent is less than training from scratch.

    ### Energy Efficiency
    The chart below shows accuracy relative to **total energy consumed**.

    For the **NCA model (orange)**, the dotted line represents the pre-training phase where accuracy on the target task is zero. The solid line represents the fine-tuning phase. If the orange line reaches the target accuracy (y-axis) at a lower energy value (x-axis) than the **Scratch model (blue)**, the pre-training was a net energy saving.
    """)
    return


@app.cell(hide_code=True)
def efficiency_metrics_summary(mo, nca_log, scratch_log):
    _nca_steps = nca_log[-1]["step"]
    _nca_energy = nca_log[-1]["energy_j"]
    _scratch_steps = scratch_log[-1]["step"]
    _scratch_energy = scratch_log[-1]["energy_j"]
    _speedup = _scratch_steps / _nca_steps
    _saving = (1 - _nca_energy / _scratch_energy) * 100

    _callout_msg = (
        (
            f"The NCA model reached **{nca_log[-1]['acc']:.1%}** accuracy in **{_nca_steps} fine-tuning steps** "
            f"vs **{_scratch_steps} steps** for scratch, a **{_speedup:.1f}× speedup**. "
            f"\n\nEven counting pre-training energy, total cost dropped by **{_saving:.1f}%**."
        )
        if _saving > 0
        else (
            f"The NCA model reached **{nca_log[-1]['acc']:.1%}** accuracy in **{_nca_steps} fine-tuning steps**, "
            f"but the total energy cost was **{abs(_saving):.1f}% higher** than training from scratch. "
            f"\n\n**Well that wasn't what we're looking for**, try reducing the NCA pre-training steps to see if we can achieve the same accuracy but with fewer steps."
        )
    )

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
                mo.md(_callout_msg),
                kind="success" if _saving > 0 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def efficiency_plots(alt, mo, nca_log, pd, pre_log: list[dict], scratch_log):
    # Prepare data
    scratch_df = pd.DataFrame([{**row, "model": "Scratch"} for row in scratch_log])

    # For NCA, we include pre-training energy to show the full lifecycle
    nca_pre_df = pd.DataFrame(
        [
            {
                "step": row["step"],
                "acc": 0,
                "energy_j": row["energy_j"],
                "model": "NCA",
            }
            for row in pre_log
        ]
    )

    nca_ft_df = pd.DataFrame([{**row, "model": "NCA"} for row in nca_log])

    # Combine everything
    compare_df = pd.concat([scratch_df, nca_pre_df, nca_ft_df], ignore_index=True)

    color = alt.Color(
        "model:N",
        scale=alt.Scale(
            domain=["Scratch", "NCA"], range=["steelblue", "darkorange"]
        ),
    )

    # PRIMARY CHART: Accuracy vs Total Energy
    energy_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("energy_j:Q", title="Total Energy Consumed (Joules)"),
            y=alt.Y(
                "acc:Q", title="Reversal Accuracy", scale=alt.Scale(domain=[0, 1])
            ),
            color=color,
            strokeDash=alt.condition(
                alt.datum.acc == 0, alt.value([3, 3]), alt.value([0])
            ),
        )
        .properties(
            title="Total Lifecycle Energy Efficiency (Pre-train + Fine-tune)",
            width=700,
            height=400,
        )
    )

    # SECONDARY CHARTS
    acc_compare = (
        alt.Chart(pd.concat([scratch_df, nca_ft_df]))
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="Fine-tune steps only"),
            y=alt.Y("acc:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            color=color,
        )
        .properties(title="Fine-tune Speedup", width=330, height=220)
    )

    pre_chart = (
        alt.Chart(pd.DataFrame(pre_log))
        .mark_line(color="purple")
        .encode(
            x=alt.X("step:Q", title="Pre-train step"),
            y=alt.Y("loss:Q", title="NCA Loss", scale=alt.Scale(zero=False)),
        )
        .properties(title="NCA Phase Convergence", width=330, height=220)
    )

    mo.vstack(
        [
            energy_compare,
            mo.md("""
    ---
    ### Secondary Metrics
    While total energy is our main constraint, we can also look at the **Fine-tune Speedup** (how many fewer steps the model needed once it saw the digits) and the **NCA Phase Convergence** (how well the model learned the synthetic patterns).
    """),
            mo.hstack([acc_compare, pre_chart], justify="center", gap="2rem"),
        ]
    )
    return


@app.cell(hide_code=True)
def part1_conclusion_md(mo):
    mo.md(r"""
    ## Conclusion: Structure Before Meaning

    Just as a child does not need to understand the story in a picture book to build the cognitive machinery for reading, a model does not need to process language to build the computational machinery for reasoning. Abstract patterns, given the right structure and complexity, are enough to give the model a meaningful head start.

    ### What's next?
    We've seen that NCA pre-training *works*, but what exactly is this "synthetic structure"? And how do we generate these patterns?

    In **Part 2**, we'll go under the hood of Neural Cellular Automata to understand the engine driving this efficiency.
    """)
    return


@app.cell(hide_code=True)
def nca_explanation_md(mo):
    mo.md(r"""
    # Part 2 What is a Neural Cellular Automaton?

    **Classical cellular automata** (like Conway's Game of Life) update every cell with the same fixed rule: look at your neighbours, apply the rule, get the next state.

    **Neural cellular automata** replace the fixed rule with a tiny neural network. Instead of hand-crafting "if three neighbours are alive, become alive", you let a small network decide the next state based on what it sees around each cell. The key insight is that the network's weights are the rule. Change the weights, and the entire character of the simulation changes.

    In this paper, those weights are randomly sampled for every sequence, meaning each training example is governed by a completely unique rule the model has never seen before and will never see again.

    Three rules run simultaneously below on a 12x12 grid, **all starting from the exact same initial state**. Each cell holds one of 10 possible states (colours). Notice how the identical starting grid diverges into completely different dynamics under each rule. Some settle into repeating cycles, others stay unpredictable. This is why the model must *infer* the hidden rule from context rather than memorize patterns.

    Try generating a new set of rules to see how it affects the grids.
    """)
    return


@app.cell(hide_code=True)
def nca_animation_widget(F, NCARule, NCA_H, NCA_STATES, NCA_W, torch):
    import anywidget
    import traitlets
    import random

    _NCA_N_RULES = 3
    _NCA_N_FRAMES = 40
    _DEFAULT_SEEDS = [
        8517247702374908565,
        4887395993740886587,
        6356219191685193974,
    ]

    _COLORS = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
    ]

    _ESM = """
    const COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ];
    const CELL = 22;

    function render({ model, el }) {
      const nRules  = model.get("n_rules");
      const H       = model.get("h");
      const W       = model.get("w");
      const nFrames = model.get("n_frames");

      let playing = false;
      let t       = 0;
      let lastTs  = 0;
      const INTERVAL = 150;
      let raf;

      /* ── DOM ── */
      const root = document.createElement("div");
      root.style.cssText = "font-family:sans-serif;user-select:none;";

      const gridsRow = document.createElement("div");
      gridsRow.style.cssText = "display:flex;gap:20px;margin-bottom:10px;";

      const canvases = [], ctxs = [];
      for (let r = 0; r < nRules; r++) {
        const wrap = document.createElement("div");
        wrap.style.textAlign = "center";
        const lbl = document.createElement("div");
        lbl.textContent = "Rule " + (r + 1);
        lbl.style.cssText = "font-size:12px;color:#666;margin-bottom:4px;";
        const cv = document.createElement("canvas");
        cv.width  = W * CELL;
        cv.height = H * CELL;
        cv.style.cssText = "border-radius:4px;display:block;";
        canvases.push(cv);
        ctxs.push(cv.getContext("2d"));
        wrap.appendChild(lbl);
        wrap.appendChild(cv);
        gridsRow.appendChild(wrap);
      }

      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap;";

      const btn = (label, extra) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = "padding:4px 10px;border:1px solid #ccc;border-radius:4px;" +
                          "cursor:pointer;background:#f5f5f5;font-size:14px;" + (extra || "");
        return b;
      };

      const stepBackBtn = btn("◀");
      const playBtn     = btn("⏸", "min-width:36px;");
      const stepFwdBtn  = btn("▶");
      const newRulesBtn = btn("New rules");

      const slider = document.createElement("input");
      slider.type  = "range";
      slider.min   = 0;
      slider.max   = nFrames - 1;
      slider.value = 0;
      slider.style.cssText = "flex:1;min-width:120px;";

      const tLabel = document.createElement("span");
      tLabel.style.cssText = "font-size:12px;color:#999;min-width:44px;";
      tLabel.textContent = "t = 0";

      controls.append(stepBackBtn, playBtn, stepFwdBtn, slider, tLabel, newRulesBtn);
      root.appendChild(gridsRow);
      root.appendChild(controls);
      el.appendChild(root);

      /* ── draw ── */
      function draw(frame) {
        const frames = model.get("frames");
        if (!frames || !frames.length) return;
        for (let r = 0; r < nRules; r++) {
          const grid = frames[r][frame];
          for (let row = 0; row < H; row++) {
            for (let col = 0; col < W; col++) {
              ctxs[r].fillStyle = COLORS[grid[row][col] % COLORS.length];
              ctxs[r].fillRect(col * CELL, row * CELL, CELL, CELL);
            }
          }
        }
        slider.value    = frame;
        tLabel.textContent = "t = " + frame;
      }

      /* ── animation loop ── */
      function loop(ts) {
        if (playing && ts - lastTs >= INTERVAL) {
          t = (t + 1) % nFrames;
          draw(t);
          lastTs = ts;
        }
        raf = requestAnimationFrame(loop);
      }

      /* ── controls ── */
      playBtn.addEventListener("click", () => {
        playing = !playing;
        playBtn.textContent = playing ? "⏸" : "▶";
        if (playing) lastTs = performance.now() - INTERVAL;
      });

      stepBackBtn.addEventListener("click", () => {
        playing = false;
        playBtn.textContent = "▶";
        t = (t - 1 + nFrames) % nFrames;
        draw(t);
      });

      stepFwdBtn.addEventListener("click", () => {
        playing = false;
        playBtn.textContent = "▶";
        t = (t + 1) % nFrames;
        draw(t);
      });

      slider.addEventListener("input", () => {
        playing = false;
        playBtn.textContent = "▶";
        t = parseInt(slider.value);
        draw(t);
      });

      newRulesBtn.addEventListener("click", () => {
        newRulesBtn.textContent = "Loading…";
        newRulesBtn.disabled = true;
        model.set("new_rules_request", model.get("new_rules_request") + 1);
        model.save_changes();
      });

      model.on("change:frames", () => {
        t = 0;
        draw(0);
        newRulesBtn.textContent = "New rules";
        newRulesBtn.disabled    = false;
      });

      draw(0);
      raf = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(raf);
    }

    export default { render };
    """


    class NCAAnimWidget(anywidget.AnyWidget):
        _esm = _ESM
        frames = traitlets.List([]).tag(sync=True)
        n_rules = traitlets.Int(_NCA_N_RULES).tag(sync=True)
        n_frames = traitlets.Int(_NCA_N_FRAMES).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)
        new_rules_request = traitlets.Int(0).tag(sync=True)
        seeds = traitlets.List(_DEFAULT_SEEDS).tag(sync=True)

        @traitlets.observe("new_rules_request")
        def _reload(self, change):
            self.seeds = [random.randint(0, 2**63 - 1) for _ in range(self.n_rules)]
            self.frames = _nca_frame_data(seeds=self.seeds)


    def _nca_frame_data(seeds=_DEFAULT_SEEDS):
        # Generate ONE shared initial grid so all rules start identically
        grid_seed = seeds[0] if seeds else 42
        torch.manual_seed(grid_seed)
        shared_init = torch.randint(0, NCA_STATES, (NCA_H, NCA_W))

        out = []
        for i in range(_NCA_N_RULES):
            seed = seeds[i] if i < len(seeds) else None
            rule = NCARule(seed=seed)

            # Every rule starts from the same grid
            grid = shared_init.clone()
            frames = [grid.tolist()]
            with torch.no_grad():
                for _ in range(_NCA_N_FRAMES - 1):
                    x = (
                        F.one_hot(grid, NCA_STATES)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    grid = rule(x).squeeze(0).argmax(0)
                    frames.append(grid.tolist())
            out.append(frames)
        return out


    nca_widget = NCAAnimWidget(frames=_nca_frame_data())
    nca_widget
    return anywidget, random, traitlets


@app.cell(hide_code=True)
def nca_step_widget(
    F,
    NCARule,
    NCA_H,
    NCA_STATES,
    NCA_W,
    anywidget,
    mo,
    torch,
    traitlets,
):
    _STEP_SHARED_ESM = """
    export default () => {
      return {
        initialize({ model }) {
          const id = setInterval(() => {
            if (model.get("playing")) {
              const grids = model.get("grids");
              if (!grids || grids.length < 2) return;

              let tidx = model.get("t_idx");
              let cidx = model.get("cell_idx");
              const w = model.get("w");
              const h = model.get("h");

              cidx++;
              if (cidx >= h * w) {
                cidx = 0;
                tidx = (tidx + 1) % (grids.length - 1);
                model.set("playing", false);
                model.save_changes();
                setTimeout(() => {
                  model.set("playing", true);
                  model.save_changes();
                }, 500);
              }

              model.set("t_idx", tidx);
              model.set("cell_idx", cidx);
              model.save_changes();
            }
          }, model.get("speed"));
          return () => clearInterval(id);
        },
        render({ model, el }) {
           const root = document.createElement("div");
           root.style.cssText = "font-family:sans-serif;user-select:none; max-width: 700px;";

           const topCard = document.createElement("div");
           topCard.style.cssText = "background: #fff; margin-bottom: 20px;";
           const topContent = document.createElement("div");
           topContent.style.cssText = "display: flex; gap: 32px; align-items: flex-start;";

           const CELL = 22;
           const W = model.get("w");
           const H = model.get("h");

           const cv = document.createElement("canvas");
           cv.width = W * CELL;
           cv.height = H * CELL;
           cv.style.cssText = "display:block;flex-shrink:0;";
           const ctx = cv.getContext("2d");

           const COLORS = [
             "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
             "#edc948","#b07aa1","#ff9da7","#9c755f",
           ];

           function draw() {
              const grids = model.get("grids");
              if (!grids) return;
              const tidx = model.get("t_idx");
              const cidx = model.get("cell_idx");
              const cur = grids[tidx];

              const hlR = Math.floor(cidx / W);
              const hlC = cidx % W;

              ctx.clearRect(0, 0, cv.width, cv.height);
              for (let r = 0; r < H; r++) {
                for (let c = 0; c < W; c++) {
                  ctx.fillStyle = COLORS[cur[r][c] % COLORS.length];
                  ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
                }
              }

              const x0 = Math.max(0, hlC - 1) * CELL;
              const y0 = Math.max(0, hlR - 1) * CELL;
              const x1 = Math.min(W, hlC + 2) * CELL;
              const y1 = Math.min(H, hlR + 2) * CELL;
              ctx.strokeStyle = "#000000";
              ctx.lineWidth = 2.5;
              ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
              ctx.fillStyle = "rgba(225,87,89,0.3)";
              ctx.fillRect(hlC * CELL, hlR * CELL, CELL, CELL);
              ctx.lineWidth = 1;
           }

           const ctrl = document.createElement("div");
           ctrl.style.cssText = "display:flex; flex-direction: column; gap: 16px; padding: 16px; background: #f9f9f9; border-radius: 8px; min-width: 200px;";

           const btnRow = document.createElement("div");
           btnRow.style.cssText = "display:flex; gap: 8px;";

           const playBtn = document.createElement("button");
           playBtn.style.cssText = "flex: 1; padding:6px 12px; border:1px solid #ccc; border-radius:4px; cursor:pointer; background:#fff; font-weight: 500;";

           const stepBtn = document.createElement("button");
           stepBtn.textContent = "Step ▶";
           stepBtn.style.cssText = "flex: 1; padding:6px 12px; border:1px solid #ccc; border-radius:4px; cursor:pointer; background:#fff; font-weight: 500;";

           btnRow.append(playBtn, stepBtn);

           const spdWrap = document.createElement("div");
           spdWrap.style.cssText = "display:flex; flex-direction: column; gap: 4px;";
           const spdLbl = document.createElement("span");
           spdLbl.style.cssText = "font-size:12px;color:#555;font-weight:500;";
           spdLbl.textContent = "Animation Speed";
           const spdSlider = document.createElement("input");
           spdSlider.type="range"; spdSlider.min=15; spdSlider.max=250; 
           spdSlider.value = 265 - model.get("speed");

           spdWrap.append(spdLbl, spdSlider);

           const infoBar = document.createElement("div");
           infoBar.style.cssText = "font-size:12px;color:#888;font-family:monospace; margin-top: auto;";

           ctrl.append(btnRow, spdWrap, infoBar);
           topContent.append(cv, ctrl);
           topCard.appendChild(topContent);
           root.appendChild(topCard);
           el.appendChild(root);

           function updateUI() {
              draw();
              playBtn.textContent = model.get("playing") ? "⏸ Pause" : "▶ Play";
              const r = Math.floor(model.get("cell_idx") / W);
              const c = model.get("cell_idx") % W;
              infoBar.textContent = `Time t=${model.get("t_idx")} | Updating cell (${r}, ${c})`;
           }

           playBtn.addEventListener("click", () => {
             model.set("playing", !model.get("playing"));
             model.save_changes();
           });

           stepBtn.addEventListener("click", () => {
             model.set("playing", false);
             let cidx = model.get("cell_idx") + 1;
             let tidx = model.get("t_idx");
             if (cidx >= H*W) { cidx=0; tidx++; }
             model.set("cell_idx", cidx);
             model.set("t_idx", tidx);
             model.save_changes();
           });

           spdSlider.addEventListener("input", () => {
             model.set("speed", 265 - parseInt(spdSlider.value));
             model.save_changes();
           });

           model.on("change:t_idx", updateUI);
           model.on("change:cell_idx", updateUI);
           model.on("change:playing", updateUI);

           updateUI();
        }
      };
    };
    """


    class NCASharedController(anywidget.AnyWidget):
        _esm = _STEP_SHARED_ESM
        grids = traitlets.List([]).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)
        t_idx = traitlets.Int(0).tag(sync=True)
        cell_idx = traitlets.Int(0).tag(sync=True)
        playing = traitlets.Bool(True).tag(sync=True)
        speed = traitlets.Int(60).tag(sync=True)


    def _nca_shared_data(n_steps=5):
        rule = NCARule()
        grid = torch.randint(0, NCA_STATES, (NCA_H, NCA_W))
        out = [grid.tolist()]
        with torch.no_grad():
            for _ in range(n_steps - 1):
                x = (
                    F.one_hot(grid, NCA_STATES)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                grid = rule(x).squeeze(0).argmax(0)
                out.append(grid.tolist())
        return out


    shared_grids = _nca_shared_data()
    nca_controller = NCASharedController(grids=shared_grids)

    get_t_idx, set_t_idx = mo.state(nca_controller.t_idx)
    get_cell_idx, set_cell_idx = mo.state(nca_controller.cell_idx)
    get_speed, set_speed = mo.state(nca_controller.speed)

    nca_controller.observe(
        lambda _: set_t_idx(nca_controller.t_idx), names=["t_idx"]
    )
    nca_controller.observe(
        lambda _: set_cell_idx(nca_controller.cell_idx), names=["cell_idx"]
    )
    nca_controller.observe(
        lambda _: set_speed(nca_controller.speed), names=["speed"]
    )

    mo.vstack(
        [
            mo.md("## How is this grid generated ?"),
            mo.md(
                "The grid is processed one cell at a time. Below is the starting grid. Use the controls to pause, step manually, or change the speed."
            ),
            nca_controller,
        ]
    )
    return get_cell_idx, get_t_idx, shared_grids


@app.cell(hide_code=True)
def nca_step_mid_widget(
    NCA_H,
    NCA_W,
    anywidget,
    get_cell_idx,
    get_t_idx,
    mo,
    shared_grids,
    traitlets,
):
    _MID_ESM = """
    const COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ];
    const NB = 40;

    function render({ model, el }) {
      const root = document.createElement("div");
      root.style.cssText = "font-family:sans-serif; max-width: 700px;";

      const midCard = document.createElement("div");
      midCard.style.cssText = "background: #fff; margin-bottom: 20px;";
      const midContent = document.createElement("div");
      midContent.style.cssText = "display: flex; align-items: center; justify-content: center; gap: 24px; padding: 20px; background: #fff;";

      const mkCv = (w, h) => {
        const cv = document.createElement("canvas");
        cv.width = w; cv.height = h;
        cv.style.cssText = "display:block;flex-shrink:0;";
        return cv;
      };

      const nbCv = mkCv(NB*3, NB*3);
      const nsCv = mkCv(NB, NB);

      const mkWrap = (cv, text) => {
        const w = document.createElement("div");
        w.style.cssText = "display: flex; flex-direction: column; align-items: center; gap: 12px;";
        const lbl = document.createElement("div");
        lbl.textContent = text;
        lbl.style.cssText = "font-size:13px; font-weight: 600; color: #555;";
        w.append(cv, lbl);
        return w;
      };

      const nbWrap = mkWrap(nbCv, "3x3 Input Patch");

      const svgNS = "http://www.w3.org/2000/svg";
      const nnSvg = document.createElementNS(svgNS, "svg");
      nnSvg.setAttribute("width", "260");
      nnSvg.setAttribute("height", "220");

      const layers = [{nodes: 9, x: 30}, {nodes: 6, x: 130}, {nodes: 1, x: 230}];
      const lines = [];
      const textNodes = { inputs: [], hidden: [], output: [] };

      for(let i=0; i<layers.length-1; i++) {
        const l1 = layers[i], l2 = layers[i+1];
        const dy1 = 220 / (l1.nodes + 1), dy2 = 220 / (l2.nodes + 1);
        for(let n1=1; n1<=l1.nodes; n1++) {
          for(let n2=1; n2<=l2.nodes; n2++) {
            const line = document.createElementNS(svgNS, "line");
            line.setAttribute("x1", l1.x);
            line.setAttribute("y1", n1 * dy1);
            line.setAttribute("x2", l2.x);
            line.setAttribute("y2", n2 * dy2);

            const weight = Math.sin(n1 * 13.3 + n2 * 17.7 + i * 23.1);
            const baseColor = weight > 0 ? `rgba(78, 121, 167, ${Math.abs(weight)*0.4 + 0.15})` : `rgba(225, 87, 89, ${Math.abs(weight)*0.4 + 0.15})`;
            const activeColor = weight > 0 ? `rgba(78, 121, 167, 1)` : `rgba(225, 87, 89, 1)`;

            line.setAttribute("stroke", baseColor);
            line.setAttribute("stroke-width", "1.5");
            line.dataset.baseColor = baseColor;
            line.dataset.activeColor = activeColor;

            nnSvg.appendChild(line);
            lines.push(line);
          }
        }
      }

      layers.forEach((l, layerIdx) => {
        const dy = 220 / (l.nodes + 1);
        for(let n=1; n<=l.nodes; n++) {
          const circle = document.createElementNS(svgNS, "circle");
          circle.setAttribute("cx", l.x);
          circle.setAttribute("cy", n * dy);
          circle.setAttribute("r", "11");
          circle.setAttribute("fill", "#dae8fc");
          circle.setAttribute("stroke", "#6c8ebf");
          circle.setAttribute("stroke-width", "1.5");
          nnSvg.appendChild(circle);

          const text = document.createElementNS(svgNS, "text");
          text.setAttribute("x", l.x);
          text.setAttribute("y", n * dy);
          text.setAttribute("dy", "0.35em");
          text.setAttribute("text-anchor", "middle");
          text.setAttribute("font-size", "10px");
          text.setAttribute("font-family", "monospace");
          text.setAttribute("font-weight", "bold");
          text.setAttribute("fill", "#333");
          nnSvg.appendChild(text);

          if (layerIdx === 0) textNodes.inputs.push(text);
          else if (layerIdx === 1) textNodes.hidden.push(text);
          else textNodes.output.push(text);
        }
      });

      const nnWrap = document.createElement("div");
      nnWrap.style.cssText = "display: flex; flex-direction: column; align-items: center; gap: 12px;";
      const nnLbl = document.createElement("div");
      nnLbl.textContent = "Learned Weights (NN)";
      nnLbl.style.cssText = "font-size:13px; font-weight: 600; color: #555;";
      nnWrap.append(nnSvg, nnLbl);

      const nsWrap = mkWrap(nsCv, "Next Cell State");

      midContent.append(nbWrap, nnWrap, nsWrap);
      midCard.appendChild(midContent);
      root.appendChild(midCard);
      el.appendChild(root);

      function draw() {
        const grids = model.get("grids");
        const tidx = model.get("t_idx");
        const cidx = model.get("cell_idx");
        const H = model.get("h");
        const W = model.get("w");

        if (!grids || !grids[tidx]) return;

        const cur = grids[tidx];
        const nxt = grids[tidx+1];
        const r = Math.floor(cidx / W);
        const c = cidx % W;

        const inputs = [];

        const ctxNb = nbCv.getContext("2d");
        ctxNb.clearRect(0,0,nbCv.width,nbCv.height);
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = ((r + dr) % H + H) % H;
            const nc = ((c + dc) % W + W) % W;
            const state = cur[nr][nc];
            inputs.push(state);

            const px = (dc + 1) * NB, py = (dr + 1) * NB;
            ctxNb.fillStyle = COLORS[state % COLORS.length];
            ctxNb.fillRect(px, py, NB, NB);

            if (dr === 0 && dc === 0) {
              ctxNb.strokeStyle = "#000000";
              ctxNb.lineWidth = 3;
              ctxNb.strokeRect(px + 1.5, py + 1.5, NB - 3, NB - 3);
              ctxNb.lineWidth = 1;
            }
            ctxNb.fillStyle = "rgba(255,255,255,0.85)";
            ctxNb.font = `bold ${Math.floor(NB * 0.4)}px sans-serif`;
            ctxNb.textAlign = "center"; ctxNb.textBaseline = "middle";
            ctxNb.fillText(state, px + NB / 2, py + NB / 2);
          }
        }

        const ctxNs = nsCv.getContext("2d");
        let outState = "";
        if (nxt && nxt[r]) {
            const state = nxt[r][c];
            outState = state;
            ctxNs.fillStyle = COLORS[state % COLORS.length];
            ctxNs.fillRect(0, 0, NB, NB);

            ctxNs.fillStyle = "rgba(255,255,255,0.9)";
            ctxNs.font = `bold ${Math.floor(NB * 0.5)}px sans-serif`;
            ctxNs.textAlign = "center"; ctxNs.textBaseline = "middle";
            ctxNs.fillText(state, NB / 2, NB / 2);
        }

        inputs.forEach((val, i) => {
            if(textNodes.inputs[i]) textNodes.inputs[i].textContent = val;
        });

        textNodes.hidden.forEach((node, i) => {
            const pseudo = Math.sin(cidx * 12.34 + tidx * 5.67 + i * 8.9) * 2 - 1;
            node.textContent = pseudo.toFixed(1);
        });

        if (textNodes.output[0]) {
            textNodes.output[0].textContent = outState;
        }

        lines.forEach(l => {
          if(Math.random() > 0.7) l.setAttribute("stroke", l.dataset.activeColor);
          else l.setAttribute("stroke", l.dataset.baseColor);
        });
        setTimeout(() => {
          lines.forEach(l => l.setAttribute("stroke", l.dataset.baseColor));
        }, 80);
      }

      model.on("change:cell_idx", draw);
      draw();
    }
    export default { render };
    """


    class NCAMidWidget(anywidget.AnyWidget):
        _esm = _MID_ESM
        grids = traitlets.List([]).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)
        t_idx = traitlets.Int(0).tag(sync=True)
        cell_idx = traitlets.Int(0).tag(sync=True)


    nca_mid = NCAMidWidget(
        grids=shared_grids, t_idx=get_t_idx(), cell_idx=get_cell_idx()
    )
    mo.vstack(
        [
            mo.md(
                """### Neural Network Rule Evaluation

    For the highlighted cell, the network takes its 3x3 neighborhood, passes it through the learned weights, and outputs a single new state."""
            ),
            nca_mid,
        ]
    )
    return


@app.cell(hide_code=True)
def nca_step_bot_widget(
    NCA_H,
    NCA_W,
    anywidget,
    get_cell_idx,
    get_t_idx,
    mo,
    shared_grids,
    traitlets,
):
    _BOT_ESM = """
    const COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ];
    const CELL = 22;

    function render({ model, el }) {
      const root = document.createElement("div");
      root.style.cssText = "font-family:sans-serif; max-width: 700px;";

      const botCard = document.createElement("div");
      botCard.style.cssText = "background: #fff; margin-bottom: 20px;";

      const H = model.get("h");
      const W = model.get("w");
      const cv = document.createElement("canvas");
      cv.width = W * CELL; cv.height = H * CELL;
      cv.style.cssText = "display:block;flex-shrink:0;";

      botCard.appendChild(cv);
      root.appendChild(botCard);
      el.appendChild(root);

      function draw() {
        const grids = model.get("grids");
        const tidx = model.get("t_idx");
        const cidx = model.get("cell_idx");
        if (!grids || !grids[tidx+1]) return;
        const nxt = grids[tidx+1];

        const ctx = cv.getContext("2d");
        ctx.clearRect(0,0,cv.width,cv.height);
        for (let r = 0; r < H; r++) {
          for (let c = 0; c < W; c++) {
            const i = r * W + c;
            if (i > cidx) {
              ctx.fillStyle = "#e0e0e0";
            } else {
              ctx.fillStyle = COLORS[nxt[r][c] % COLORS.length];
            }
            ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
          }
        }
      }

      model.on("change:cell_idx", draw);
      draw();
    }
    export default { render };
    """


    class NCABotWidget(anywidget.AnyWidget):
        _esm = _BOT_ESM
        grids = traitlets.List([]).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)
        t_idx = traitlets.Int(0).tag(sync=True)
        cell_idx = traitlets.Int(0).tag(sync=True)


    nca_bot = NCABotWidget(
        grids=shared_grids, t_idx=get_t_idx(), cell_idx=get_cell_idx()
    )
    mo.vstack(
        [
            mo.md(
                """### Next Grid State
    The newly computed state is placed into the next timeframe's grid. Once all cells update, the process repeats."""
            ),
            nca_bot,
        ]
    )
    return


@app.cell(hide_code=True)
def complexity_intro_md(mo):
    mo.md(r"""
    ---
    ## What do we mean by "complexity"?

    As you just saw, some grids settle into predictable repeating patterns while others stay chaotic and hard to anticipate. That difference is what the paper calls complexity, and it turns out to be one of the most important levers in the whole approach.

    The paper filters NCA rules by **gzip compressibility**, a practical proxy for how much structure a sequence contains.
    gzip finds repeated patterns and encodes them efficiently. A file that compresses to 30% of its original size is simple: it is full of repetition. A file that barely compresses is complex: every part carries new information.

    $$\text{gzip ratio} = \frac{\text{compressed size}}{\text{raw size}} \times 100\%$$

    - Code is predictable: keywords, indentation, brackets repeat constantly.
    - Natural language is harder to predict; vocabulary and structure vary far more.

    The grids you just watched follow the same logic: the ones that settled into cycles would compress easily, the chaotic ones would not.
    **Edit the examples below** to see how different text affects the ratio.
    """)
    return


@app.cell(hide_code=True)
def complexity_stats(complexity_code_input, complexity_text_input, mo):
    import gzip as _gzip


    def _gz_ratio(text):
        raw = text.encode("utf-8")
        if len(raw) < 20:
            return None
        return round(len(_gzip.compress(raw, compresslevel=9)) / len(raw) * 100, 1)


    _code_ratio = _gz_ratio(complexity_code_input.value)
    _text_ratio = _gz_ratio(complexity_text_input.value)

    mo.hstack(
        [
            mo.stat(
                value=f"{_code_ratio}%" if _code_ratio else "N/A",
                label="Code gzip ratio",
            ),
            mo.stat(
                value=f"{_text_ratio}%" if _text_ratio else "N/A",
                label="Text gzip ratio",
            ),
        ],
        justify="space-around",
        gap="2rem",
    )
    return


@app.cell(hide_code=True)
def complexity_input_widgets(mo):
    _default_code = """def forward(x, weight, bias):
        return x @ weight.T + bias

    def relu(x):
        return max(0, x)

    def sigmoid(x):
        return 1 / (1 + exp(-x))

    def mse_loss(pred, target):
        return mean((pred - target) ** 2)

    def cross_entropy(pred, target):
        return -sum(target * log(pred))

    def accuracy(pred, target):
        return mean(argmax(pred) == argmax(target))
    """

    _default_text = """Notebooks of the past were graveyards of stale state, cells run out of order, outputs left to their fate.
    Then Marimo arrived and rewrote the deal, reactive by nature, every output is real.
    Change a slider, a value, a line in your code, the whole notebook updates, carries its own load.
    So here is to Marimo, the notebook grown up, patient and reactive, never running corrupt.
    """

    complexity_code_input = mo.ui.code_editor(
        value=_default_code,
        language="python",
        label="Code",
        min_height=200,
        debounce=True,
    )

    complexity_text_input = mo.ui.text_area(
        value=_default_text,
        label="Natural language",
        rows=18,
        full_width=True,
        debounce=True,
    )

    mo.hstack(
        [complexity_code_input, complexity_text_input],
        gap="1rem",
        justify="space-around",
    )
    return complexity_code_input, complexity_text_input


@app.cell(hide_code=True)
def complexity_ratio_plot(
    alt,
    complexity_code_input,
    complexity_text_input,
    mo,
    pd,
):
    import gzip as _gzip


    def _gz(text):
        raw = text.encode("utf-8")
        if len(raw) < 20:
            return None
        return round(len(_gzip.compress(raw, compresslevel=9)) / len(raw) * 100, 1)


    _code_r = _gz(complexity_code_input.value)
    _text_r = _gz(complexity_text_input.value)

    _rows = []
    if _code_r is not None:
        _rows.append({"label": "Your code", "ratio": _code_r, "group": "yours"})
    if _text_r is not None:
        _rows.append({"label": "Your text", "ratio": _text_r, "group": "yours"})

    _rows += [
        {"label": "CodeParrot corpus", "ratio": 32, "group": "corpus"},
        {"label": "OpenWebMath corpus", "ratio": 58, "group": "corpus"},
        {"label": "OpenWebText corpus", "ratio": 70, "group": "corpus"},
        {"label": "NCA 50%+ band", "ratio": 50, "group": "nca"},
    ]

    _df = pd.DataFrame(_rows)

    _colors = alt.Color(
        "group:N",
        scale=alt.Scale(
            domain=["yours", "corpus", "nca"],
            range=["#f28e2b", "#76b7b2", "#e15759"],
        ),
        legend=alt.Legend(title=None, orient="bottom"),
    )

    _bars = (
        alt.Chart(_df)
        .mark_bar(height=20, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "ratio:Q",
                title="gzip ratio  (lower = more compressible = simpler)",
                scale=alt.Scale(domain=[0, 105]),
            ),
            y=alt.Y("label:N", sort=None, title=None),
            color=_colors,
            tooltip=[
                "label:N",
                alt.Tooltip("ratio:Q", title="gzip %", format=".1f"),
            ],
        )
    )

    _rule = (
        alt.Chart(pd.DataFrame({"x": [50]}))
        .mark_rule(color="#e15759", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )

    _text_mark = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [51],
                    "y": ["CodeParrot corpus"],
                    "label": ["← NCA filter threshold"],
                }
            )
        )
        .mark_text(align="left", fontSize=10, color="#e15759")
        .encode(x="x:Q", y=alt.Y("y:N"), text="label:N")
    )

    _chart = (_bars + _rule + _text_mark).properties(width=520, height=175)

    _stat_lines = []
    if _code_r:
        _stat_lines.append(f"**Your code:** {_code_r}%")
    if _text_r:
        _stat_lines.append(f"**Your text:** {_text_r}%")

    mo.vstack(
        [
            mo.md("""### How do your inputs compare to LLM training data?

    The chart below plots the gzip ratio of your custom snippets against three massive datasets commonly used to train language models:

    - **CodeParrot:** A dataset of Python code. Like your code snippet, it compresses very efficiently because of strict syntax and repeated keywords.
    - **OpenWebMath:** Mathematical web pages and formulas. It has more structure than regular text but more entropy than code.
    - **OpenWebText:** A massive corpus of general internet text (Reddit links, articles, etc.). Natural language is full of varied vocabulary and irregular sentence structures, making it much harder to compress.

    """),
            _chart,
        ],
    )
    return


@app.cell(hide_code=True)
def nca_complexity_md(mo):
    mo.md(r"""
    ## NCA sequence complexity

    The same logic applies directly to NCA grids. Just as your code snippet compressed more easily than your natural language snippet, NCA sequences have their own gzip ratio. Serialize the grid states over time into a byte stream, and you can measure exactly how much structure they contain.

    Most random rules fall into one of two failure modes:

    | Regime | What happens | gzip ratio |
    |---|---|---|
    | **Simple / periodic** | Cells lock into a repeating checkerboard or solid colour. Every frame is nearly identical. | ~20–35% |
    | **Chaotic / random** | Each state is essentially independent noise. The grid never settles into any pattern. | ~85–95% |

    Neither extreme is useful for training. A model that sees only repetition learns nothing new after the first few frames. A model that sees only noise has nothing to learn at all.

    The paper filters rules to the 50-75% band, sitting between those two failure modes. Not trivial, not random, but structured at an intermediate scale. That is where learnable sequential structure lives, and as you saw in the chart above, it sits in roughly the same complexity range as natural language.

    Hit **Resample** below to draw fresh rules from each band and see the difference.
    """)
    return


@app.cell(hide_code=True)
def nca_resample_button(mo):
    nca_resample_btn = mo.ui.run_button(label="Resample rules")
    nca_resample_btn
    return (nca_resample_btn,)


@app.cell(hide_code=True)
def nca_complexity_bands_plot(
    F,
    NCA_H,
    NCA_STATES,
    NCA_W,
    mo,
    nca_resample_btn,
    torch,
):
    nca_resample_btn  # reactive dependency

    import matplotlib.pyplot as _plt
    import matplotlib.gridspec as _gs
    import gzip as _gzip3
    import numpy as _np

    _N_SAMPLE = 128

    _BANDS = [
        ("Simple / periodic", 0, 30, "#76b7b2"),
        ("Structured (paper target)", 30, 80, "#59a14f"),
        ("Chaotic / random", 80, 101, "#e15759"),
    ]


    class _NCARuleVis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                NCA_STATES, 4, 3, padding=1, padding_mode="circular"
            )
            self.expand = torch.nn.Conv2d(4, 16, 1)
            self.out = torch.nn.Conv2d(16, NCA_STATES, 1)
            for layer in [self.conv, self.expand, self.out]:
                torch.nn.init.normal_(layer.weight, std=0.1)
                torch.nn.init.zeros_(layer.bias)

        def forward(self, x):
            return self.out(torch.relu(self.expand(self.conv(x))))


    _PATCH = 3
    _POWERS = _np.array(
        [NCA_STATES**i for i in range(_PATCH * _PATCH)], dtype=_np.int64
    )


    def _patch_gzip(frames):
        """Encode each 3×3 patch as a unique int32 token, then gzip.
        Gives full 0–100% spread vs raw bytes which are capped at ~48%
        due to gzip's alphabet model compressing small-value bytes regardless."""
        result = []
        for frame in frames:
            arr = frame.numpy()
            for pr in range(NCA_H // _PATCH):
                for pc in range(NCA_W // _PATCH):
                    patch = arr[
                        pr * _PATCH : (pr + 1) * _PATCH,
                        pc * _PATCH : (pc + 1) * _PATCH,
                    ].flatten()
                    result.append(int(_np.dot(patch, _POWERS)))
        data = _np.array(result, dtype=_np.int32).tobytes()
        return round(
            len(_gzip3.compress(data, compresslevel=9)) / len(data) * 100, 1
        )


    def _run_rule(n_steps=32):
        rule = _NCARuleVis()
        grid = torch.randint(0, NCA_STATES, (NCA_H, NCA_W))
        frames = [grid.clone()]
        with torch.no_grad():
            for _ in range(n_steps - 1):
                x = (
                    F.one_hot(grid, NCA_STATES)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                grid = rule(x).squeeze(0).argmax(0)
                frames.append(grid.clone())
        return _patch_gzip(frames), frames


    _pool = [_run_rule() for _ in range(_N_SAMPLE)]


    def _best_for_band(lo, hi):
        mid = (lo + hi) / 2
        in_band = [(r, f) for r, f in _pool if lo <= r < hi]
        candidates = in_band if in_band else _pool
        ratio, frames = min(candidates, key=lambda x: abs(x[0] - mid))
        return ratio, frames, bool(in_band)


    _results = [
        (_label, _lo, _hi, _color, *_best_for_band(_lo, _hi))
        for _label, _lo, _hi, _color in _BANDS
    ]

    _SNAP_T = [0, 4, 8, 15]
    _CMAP = _plt.colormaps.get_cmap("tab10").resampled(NCA_STATES)

    _fig = _plt.figure(figsize=(9, 5.5))
    _fig.patch.set_facecolor("white")
    _outer = _gs.GridSpec(
        3, 2, figure=_fig, hspace=0.6, wspace=0.08, width_ratios=[0.22, 0.78]
    )

    for _i, (_label, _lo, _hi, _color, _ratio, _frames, _in_band) in enumerate(
        _results
    ):
        _ax_lbl = _fig.add_subplot(_outer[_i, 0])
        _ax_lbl.axis("off")
        _ax_lbl.text(
            0.95,
            0.65,
            _label,
            transform=_ax_lbl.transAxes,
            ha="right",
            va="center",
            fontsize=9,
            color=_color,
            fontweight="bold",
        )
        _band_str = str(_lo) + "–" + str(_hi) + "% band"
        _ax_lbl.text(
            0.95,
            0.38,
            _band_str,
            transform=_ax_lbl.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color="#aaa",
        )
        _suffix = "" if _in_band else " *"
        _ax_lbl.text(
            0.95,
            0.12,
            "gzip = " + str(_ratio) + "%" + _suffix,
            transform=_ax_lbl.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            color=_color,
            fontweight="bold",
            family="monospace",
        )

        _inner = _gs.GridSpecFromSubplotSpec(
            1, len(_SNAP_T), subplot_spec=_outer[_i, 1], wspace=0.05
        )
        for _j, _t in enumerate(_SNAP_T):
            _ax = _fig.add_subplot(_inner[_j])
            _ax.imshow(
                _frames[_t].numpy(),
                cmap=_CMAP,
                vmin=0,
                vmax=NCA_STATES - 1,
                interpolation="nearest",
            )
            _ax.set_title("t = " + str(_t), fontsize=8, color="#888", pad=3)
            _ax.axis("off")

    _fig.suptitle(
        "Best rule from each complexity band  (sampled "
        + str(_N_SAMPLE)
        + " rules)",
        fontsize=10,
        y=1.02,
        color="#444",
    )

    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def alphabet_size_intro(mo):
    mo.md(r"""
    ### Alphabet Size: The "Vocabulary" of the Universe

    In the previous section, we saw that we can filter rules based on their complexity. But what determines the "default" complexity of the whole universe?

    In NCA, this is controlled by the **Alphabet Size** ($n$), the number of possible states (or colors) each cell can take. This acts as the "vocabulary" of the simulation.

    The paper investigates three sizes: $n=2, 10, 15$. They discovered a fascinating trade-off:
    - **Small Alphabets ($n=2$):** The rule space is smaller. While individual sequences might seem "simpler," the patterns are more distinct and scale better as models grow.
    - **Large Alphabets ($n=15$):** The rule space is exponentially more expressive, but it becomes much harder to find "structured" rules; most random samples at high $n$ fall straight into chaotic noise.

    Use the slider below to see how the statistical "center of gravity" of the universe shifts from structure toward chaos as you increase the number of colors.
    """)
    return


@app.cell(hide_code=True)
def complexity_precompute(F, NCARule, gzip, np, torch):
    def _compute_dataset():
        _dataset = {}
        _n_values = list(range(2, 18, 2))
        _samples = 128
        _temp_h, _temp_w = 12, 12
        _PATCH_SZ = 3

        for _n in _n_values:
            _ratios = []
            _sample_grids = []
            for _i in range(_samples):
                _rule = NCARule(n_states=_n)
                _grid = torch.randint(0, _n, (_temp_h, _temp_w))
                _frames = [_grid.tolist()]
                with torch.no_grad():
                    for _ in range(4):
                        _x = (
                            F.one_hot(_grid, _n)
                            .float()
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                        )
                        _grid = _rule(_x).squeeze(0).argmax(0)
                        _frames.append(_grid.tolist())

                _POWERS = np.array(
                    [_n**i for i in range(_PATCH_SZ * _PATCH_SZ)], dtype=np.int64
                )
                _result = []
                for _f in _frames:
                    _arr = np.array(_f)
                    for _pr in range(_temp_h // _PATCH_SZ):
                        for _pc in range(_temp_w // _PATCH_SZ):
                            _p = _arr[
                                _pr * _PATCH_SZ : (_pr + 1) * _PATCH_SZ,
                                _pc * _PATCH_SZ : (_pc + 1) * _PATCH_SZ,
                            ].flatten()
                            _result.append(int(np.dot(_p, _POWERS)))
                # Use int64 to avoid overflow with large n
                _data = np.array(_result, dtype=np.int64).tobytes()
                _ratio = round(len(gzip.compress(_data)) / len(_data) * 100, 1)
                _ratios.append(_ratio)
                if _i < 4:
                    _sample_grids.append(_frames[-1])
            _dataset[_n] = {"ratios": _ratios, "grids": _sample_grids}
        return _dataset


    complexity_cache = _compute_dataset()
    return (complexity_cache,)


@app.cell(hide_code=True)
def alphabet_slider(mo):
    n_slider = mo.ui.slider(
        start=2, stop=16, step=2, value=2, label="Alphabet Size (n)"
    )
    n_slider
    return (n_slider,)


@app.cell(hide_code=True)
def alphabet_complexity_logic(alt, complexity_cache, mo, n_slider, pd, random):
    # Pull from cache
    _data_entry = complexity_cache[n_slider.value]
    _ratios = _data_entry["ratios"]
    _grid = _data_entry["grids"][random.randint(0, len(_data_entry["grids"]) - 1)]

    # Fixed Bin Histogram
    _df = pd.DataFrame({"ratio": _ratios})
    _hist = (
        alt.Chart(_df)
        .mark_bar(color="#4e79a7", opacity=0.7)
        .encode(
            x=alt.X(
                "ratio:Q",
                bin=alt.Bin(step=5, extent=[0, 100]),
                title="Complexity (Gzip Ratio %)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("count()", title="Frequency", scale=alt.Scale(domain=[0, 100])),
        )
        .properties(
            width=450,
            height=220,
            title=f"Alphabet n={n_slider.value} Complexity Distribution",
        )
    )

    _bands = (
        alt.Chart(pd.DataFrame({"x": [30, 80]}))
        .mark_rule(strokeDash=[4, 4], color="red", size=2)
        .encode(x="x:Q")
    )


    def _render_single_grid(grid):
        _colors = [
            "#4e79a7",
            "#f28e2b",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc948",
            "#b07aa1",
            "#ff9da7",
            "#9c755f",
            "#d4d4d4",
            "#000",
            "#333",
            "#555",
            "#777",
            "#999",
            "#bbb",
        ]
        _rows = "".join(
            [
                f'<div style="line-height:0;">'
                + "".join(
                    [
                        f'<div style="width:14px; height:14px; background:{_colors[v % len(_colors)]}; display:inline-block; border:0.1px solid #eee;"></div>'
                        for v in r
                    ]
                )
                + "</div>"
                for r in grid
            ]
        )
        return mo.Html(
            f'<div style="border:1px solid #444; padding:4px; background:#fff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">{_rows}</div>'
        )


    _gallery = mo.hstack([_render_single_grid(g) for g in [_grid]], justify="start")

    # Dynamic Insight using n_slider.value
    _insight = mo.md(f"""
    **The Insight:** 
    {
        f"At **n={n_slider.value}**, rules are simpler and patterns are more distinct. The Transformer can easily learn the 'physics' of the system."
        if n_slider.value < 6
        else f"At **n={n_slider.value}**, we have shifted deep into 'Large Vocabulary' territory. Most random rules produce chaotic noise (high Gzip ratios), making it harder for the model to find structure."
    }
    """)

    _matters_md = mo.md(r"""
    ### Why this matters for the Paper
    $n=2$ (the smallest alphabet) actually showed the best long-term scaling. Even though it's "simpler," its structural clarity makes it a more efficient "picture book" for the Transformer's attention heads to learn from compared to the noisy soup of $n=15$.
    """)

    mo.vstack(
        [
            mo.hstack(
                [
                    _hist + _bands,
                    mo.vstack(
                        [mo.md("### Typical Sample"), _gallery, _insight],
                        align="center",
                    ),
                ],
                gap=4,
                align="start",
            ),
            _matters_md,
        ]
    )
    return


@app.cell(hide_code=True)
def nca_token_intro_md(mo):
    mo.md(r"""
    ## From Grids to Tokens: How the Model Actually Sees NCA Data

    We've built up a picture of what NCA dynamics look like and how their complexity is controlled. But a Transformer doesn't see colorful grids. It sees **sequences of integers**. So how do we turn a 2D evolving grid into something a language model can chew on?

    The answer is surprisingly simple: the same way language models process text. Text gets split into tokens (short chunks mapped to integer IDs), and the model learns to predict the next token from context. We do exactly the same thing with NCA grids, by slicing each frame into small patches and encoding each patch as a single token.
    """)
    return


@app.cell(hide_code=True)
def timestep_unroller(
    F,
    NCARule,
    NCA_H,
    NCA_STATES,
    NCA_W,
    anywidget,
    torch,
    traitlets,
):
    _UNROLLER_ESM = """
    const COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ];
    const CELL = 18;
    const PATCH = 2;

    export default {
        render({ model, el }) {
            const root = document.createElement("div");
            root.style.cssText = "font-family:sans-serif; user-select:none; max-width: 1000px; padding: 20px; background: #fff;";

            const container = document.createElement("div");
            container.style.cssText = "display: flex; gap: 30px; align-items: flex-start;";

            const col1 = document.createElement("div");
            col1.style.cssText = "display: flex; flex-direction: column; align-items: center; gap: 10px; width: 230px;";
            const title1 = document.createElement("div");
            title1.textContent = "1. 12x12 Grid";
            title1.style.cssText = "font-size: 14px; font-weight: 600; color: #444; width: 100%;";

            const W = model.get("w");
            const H = model.get("h");
            const canvas = document.createElement("canvas");
            canvas.width = W * CELL;
            canvas.height = H * CELL;
            canvas.style.cssText = "border: 1px solid #eee; border-radius: 4px; display: block;";
            const ctx = canvas.getContext("2d");

            col1.append(title1, canvas);

            const col2 = document.createElement("div");
            col2.style.cssText = "display: flex; flex-direction: column; gap: 10px; width: 350px;";
            const title2 = document.createElement("div");
            title2.textContent = "2. Visual Patches (36 total)";
            title2.style.cssText = "font-size: 14px; font-weight: 600; color: #444;";

            const seqContainer = document.createElement("div");
            seqContainer.style.cssText = "display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; padding: 10px; background: #fdfdfd; border-radius: 6px; height: 320px; overflow-y: auto; align-content: start; border: 1px solid #f0f0f0;";

            col2.append(title2, seqContainer);

            const col3 = document.createElement("div");
            col3.style.cssText = "display: flex; flex-direction: column; gap: 10px; flex: 1;";
            const title3 = document.createElement("div");
            title3.textContent = "3. Serialized Output";
            title3.style.cssText = "font-size: 14px; font-weight: 600; color: #444;";

            const textOutput = document.createElement("div");
            textOutput.style.cssText = "padding: 12px; background: #f4f4f4; border-radius: 6px; font-family: 'Fira Code', 'Courier New', monospace; font-size: 12px; color: #333; line-height: 1.6; height: 320px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid #eee;";
            textOutput.textContent = "<grid> ";

            col3.append(title3, textOutput);

            container.append(col1, col2, col3);

            const controls = document.createElement("div");
            controls.style.cssText = "margin-top: 20px; display: flex; gap: 15px; align-items: center;";

            const stepBtn = document.createElement("button");
            stepBtn.textContent = "Process Next Patch";
            stepBtn.style.cssText = "padding: 8px 20px; background: #4e79a7; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;";

            const resetBtn = document.createElement("button");
            resetBtn.textContent = "Reset";
            resetBtn.style.cssText = "padding: 8px 16px; background: #fff; border: 1px solid #ccc; border-radius: 4px; cursor: pointer;";

            const newRuleBtn = document.createElement("button");
            newRuleBtn.textContent = "New Rule (Structured)";
            newRuleBtn.style.cssText = "padding: 8px 16px; background: #59a14f; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;";

            const progressText = document.createElement("div");
            progressText.style.cssText = "font-size: 13px; color: #666; font-family: monospace;";

            controls.append(stepBtn, resetBtn, newRuleBtn, progressText);
            root.append(container, controls);
            el.append(root);

            let patchIdx = 0;
            const totalPatches = (W/PATCH) * (H/PATCH);
            let frames = model.get("frames");
            let grid = frames[0];

            function drawGrid() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                for (let r = 0; r < H; r++) {
                    for (let c = 0; c < W; c++) {
                        ctx.fillStyle = COLORS[grid[r][c] % COLORS.length];
                        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
                    }
                }
                if (patchIdx < totalPatches) {
                    const pr = Math.floor(patchIdx / (W/PATCH));
                    const pc = patchIdx % (W/PATCH);
                    ctx.strokeStyle = "#000000";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(pc * PATCH * CELL, pr * PATCH * CELL, PATCH * CELL, PATCH * CELL);
                }
            }

            function addToken() {
                if (patchIdx >= totalPatches) return;
                const pr = Math.floor(patchIdx / (W/PATCH));
                const pc = patchIdx % (W/PATCH);
                let pVals = [];
                for(let dr=0; dr<PATCH; dr++) {
                    for(let dc=0; dc<PATCH; dc++) {
                        pVals.push(grid[pr*PATCH + dr][pc*PATCH + dc]);
                    }
                }
                const tokenId = pVals.join("");
                const tokenBox = document.createElement("div");
                tokenBox.style.cssText = "display: flex; align-items: center; gap: 6px; background: #fff; border: 1px solid #eee; border-radius: 4px; padding: 4px; animation: pop 0.2s ease-out;";
                const miniGrid = document.createElement("div");
                miniGrid.style.cssText = `display: grid; grid-template-columns: repeat(${PATCH}, 1fr); gap: 1px;`;
                pVals.forEach(v => {
                    const dot = document.createElement("div");
                    dot.style.cssText = `width: 6px; height: 6px; background: ${COLORS[v % COLORS.length]}; border-radius: 1px;`;
                    miniGrid.append(dot);
                });
                const tokenLabel = document.createElement("div");
                tokenLabel.textContent = tokenId;
                tokenLabel.style.cssText = "font-size: 10px; color: #444; font-family: monospace; font-weight: 600;";
                tokenBox.append(miniGrid, tokenLabel);
                seqContainer.append(tokenBox);
                seqContainer.scrollTop = seqContainer.scrollHeight;
                textOutput.textContent += tokenId + " ";
                if (patchIdx === totalPatches - 1) { textOutput.textContent += "</grid>"; }
                textOutput.scrollTop = textOutput.scrollHeight;
                patchIdx++;
                progressText.textContent = `Patch ${patchIdx}/${totalPatches}`;
                drawGrid();
            }

            stepBtn.addEventListener("click", addToken);
            resetBtn.addEventListener("click", () => {
                patchIdx = 0; seqContainer.innerHTML = ""; textOutput.textContent = "<grid> "; progressText.textContent = ""; drawGrid();
            });
            newRuleBtn.addEventListener("click", () => {
                newRuleBtn.textContent = "Searching...";
                newRuleBtn.disabled = true;
                model.set("new_rule_request", model.get("new_rule_request") + 1);
                model.save_changes();
            });
            model.on("change:frames", () => {
                frames = model.get("frames"); grid = frames[0];
                patchIdx = 0; seqContainer.innerHTML = ""; textOutput.textContent = "<grid> "; 
                progressText.textContent = `New Rule Loaded`;
                newRuleBtn.textContent = "New Rule (Structured)";
                newRuleBtn.disabled = false;
                drawGrid();
            });
            const style = document.createElement("style");
            style.textContent = `@keyframes pop { 0% { transform: scale(0.9); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }`;
            document.head.append(style);
            drawGrid();
        }
    }
    """

    import numpy as _np
    import gzip as _gzip


    def _patch_gzip_filter(frames):
        _PATCH_SZ = 3
        _POWERS_FILTER = _np.array(
            [NCA_STATES**i for i in range(_PATCH_SZ * _PATCH_SZ)], dtype=_np.int64
        )
        result = []
        for frame in frames:
            arr = _np.array(frame)
            for pr in range(NCA_H // _PATCH_SZ):
                for pc in range(NCA_W // _PATCH_SZ):
                    patch = arr[
                        pr * _PATCH_SZ : (pr + 1) * _PATCH_SZ,
                        pc * _PATCH_SZ : (pc + 1) * _PATCH_SZ,
                    ].flatten()
                    result.append(int(_np.dot(patch, _POWERS_FILTER)))
        data = _np.array(result, dtype=_np.int32).tobytes()
        return round(
            len(_gzip.compress(data, compresslevel=9)) / len(data) * 100, 1
        )


    def _generate_structured_trajectory(n_steps=15, lo=40, hi=75):
        for _ in range(50):
            rule = NCARule()
            grid = torch.randint(0, NCA_STATES, (NCA_H, NCA_W))
            frames = [grid.tolist()]
            with torch.no_grad():
                for _s in range(n_steps - 1):
                    x = (
                        F.one_hot(grid, NCA_STATES)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    grid = rule(x).squeeze(0).argmax(0)
                    frames.append(grid.tolist())
            ratio = _patch_gzip_filter(frames)
            if lo <= ratio <= hi:
                return ratio, frames
        return ratio, frames


    class SharedNCAState(anywidget.AnyWidget):
        _esm = _UNROLLER_ESM
        frames = traitlets.List([]).tag(sync=True)
        ratio = traitlets.Float(0.0).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)
        new_rule_request = traitlets.Int(0).tag(sync=True)

        @traitlets.observe("new_rule_request")
        def _handle_new_rule(self, change):
            r, f = _generate_structured_trajectory()
            self.ratio = r
            self.frames = f


    _init_r, _init_f = _generate_structured_trajectory()
    shared_nca = SharedNCAState(frames=_init_f, ratio=_init_r)
    shared_nca
    return (shared_nca,)


@app.cell(hide_code=True)
def unroller_to_evolution_bridge(mo):
    mo.md(r"""
    That was a single timestep. But NCA rules run for many steps, and the model needs to see the full trajectory to learn the underlying rule. Watch below as each new frame of the NCA's evolution gets appended to a growing token stream, building the spatiotemporal history that the model will learn to predict.
    """)
    return


@app.cell(hide_code=True)
def sequence_evolution_anim(
    NCA_H,
    NCA_W,
    anywidget,
    mo,
    shared_nca,
    traitlets,
):
    _SEQ_ANIM_ESM = """
    const COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ];
    const CELL = 10;
    const PATCH = 2;

    export default {
        render({ model, el }) {
            const root = document.createElement("div");
            root.style.cssText = "font-family:sans-serif; user-select:none; max-width: 1000px; padding: 20px; background: #fff;";
            const container = document.createElement("div");
            container.style.cssText = "display: flex; gap: 30px; align-items: stretch;";
            const leftCol = document.createElement("div");
            leftCol.style.cssText = "display: flex; flex-direction: column; gap: 10px; width: 150px;";
            const title1 = document.createElement("div");
            title1.textContent = "NCA Evolution";
            title1.style.cssText = "font-size: 14px; font-weight: 600; color: #444;";
            const canvas = document.createElement("canvas");
            const W = model.get("w"); const H = model.get("h");
            canvas.width = W * CELL; canvas.height = H * CELL;
            canvas.style.cssText = "border: 1px solid #eee; border-radius: 4px; display: block;";
            const ctx = canvas.getContext("2d");
            const stepInfo = document.createElement("div");
            stepInfo.style.cssText = "font-size: 12px; color: #666; font-family: monospace;";
            leftCol.append(title1, canvas, stepInfo);
            const rightCol = document.createElement("div");
            rightCol.style.cssText = "display: flex; flex-direction: column; gap: 10px; flex: 1;";
            const title2 = document.createElement("div");
            title2.textContent = "Flattened Token Stream (Next-Token Prediction Target)";
            title2.style.cssText = "font-size: 14px; font-weight: 600; color: #444;";
            const streamContainer = document.createElement("div");
            streamContainer.style.cssText = "padding: 15px; background: #f4f4f4; color: #333; border-radius: 6px; font-family: 'Fira Code', 'Courier New', monospace; font-size: 11px; height: 300px; overflow-y: auto; line-height: 1.8; word-break: break-all; border: 1px solid #eee;";
            rightCol.append(title2, streamContainer);
            container.append(leftCol, rightCol);
            const controls = document.createElement("div");
            controls.style.cssText = "margin-top: 20px; display: flex; gap: 15px; align-items: center;";
            const playBtn = document.createElement("button");
            playBtn.textContent = "▶ Start Evolution";
            playBtn.style.cssText = "padding: 8px 20px; background: #4e79a7; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;";
            const resetBtn = document.createElement("button");
            resetBtn.textContent = "Reset";
            resetBtn.style.cssText = "padding: 8px 16px; background: #fff; border: 1px solid #ccc; border-radius: 4px; cursor: pointer;";
            controls.append(playBtn, resetBtn);
            root.append(container, controls);
            el.append(root);
            let frameIdx = 0; let isPlaying = false; let interval;
            let frames = model.get("frames");
            function drawFrame(idx) {
                const grid = frames[idx];
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                for (let r = 0; r < H; r++) { for (let c = 0; c < W; c++) { ctx.fillStyle = COLORS[grid[r][c] % COLORS.length]; ctx.fillRect(c * CELL, r * CELL, CELL, CELL); } }
                stepInfo.textContent = `Timestep t = ${idx}`;
            }
            function getFrameTokens(idx) {
                const grid = frames[idx]; let tokens = [];
                for (let pr = 0; pr < H / PATCH; pr++) { for (let pc = 0; pc < W / PATCH; pc++) {
                        let p = ""; for (let dr = 0; dr < PATCH; dr++) { for (let dc = 0; dc < PATCH; dc++) { p += grid[pr * PATCH + dr][pc * PATCH + dc]; } }
                        tokens.push(p);
                } }
                return tokens;
            }
            function appendTokens(idx) {
                const tokens = getFrameTokens(idx); const block = document.createElement("span");
                block.style.cssText = "display: inline; margin-right: 8px; animation: fadein 0.3s ease-out;";
                const head = document.createElement("span"); head.textContent = "<grid> "; head.style.color = "#4e79a7"; head.style.fontWeight = "bold";
                const tail = document.createElement("span"); tail.textContent = "</grid> "; tail.style.color = "#4e79a7"; tail.style.fontWeight = "bold";
                block.append(head);
                tokens.forEach((t, i) => { const s = document.createElement("span"); s.textContent = t + " "; s.style.color = "#333"; block.append(s); });
                block.append(tail);
                streamContainer.append(block); streamContainer.scrollTop = streamContainer.scrollHeight;
            }
            function play() {
                if (frameIdx >= frames.length) { stop(); return; }
                drawFrame(frameIdx); appendTokens(frameIdx); frameIdx++;
            }
            function stop() { isPlaying = false; clearInterval(interval); playBtn.textContent = "▶ Resume Evolution"; }
            playBtn.addEventListener("click", () => {
                if (isPlaying) { stop(); } else { isPlaying = true; playBtn.textContent = "Pause"; interval = setInterval(play, 400); }
            });
            resetBtn.addEventListener("click", () => {
                stop(); frameIdx = 0; streamContainer.innerHTML = ""; drawFrame(0); playBtn.textContent = "▶ Start Evolution";
            });
            model.on("change:frames", () => { frames = model.get("frames"); frameIdx = 0; streamContainer.innerHTML = ""; drawFrame(0); stop(); playBtn.textContent = "▶ Start Evolution"; });
            const style = document.createElement("style");
            style.textContent = `@keyframes fadein { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }`;
            document.head.append(style);
            drawFrame(0);
        }
    }
    """


    class SequenceEvolutionAnim(anywidget.AnyWidget):
        _esm = _SEQ_ANIM_ESM
        frames = traitlets.List([]).tag(sync=True)
        h = traitlets.Int(NCA_H).tag(sync=True)
        w = traitlets.Int(NCA_W).tag(sync=True)


    seq_evolution_widget = SequenceEvolutionAnim(
        frames=shared_nca.frames, h=shared_nca.h, w=shared_nca.w
    )

    shared_nca.observe(
        lambda change: setattr(seq_evolution_widget, "frames", change.new),
        names=["frames"],
    )

    mo.vstack(
        [
            mo.md(r"""
    ### Multi-Step Serialization

    One timestep of the NCA produces **36 tokens**. Run for $T$ steps and you
    get a sequence of $36T$ integers, exactly the shape a language model expects.

    Watch below as each new "frame" of the NCA's evolution is appended to the 
    continuous stream of tokens, building the spatial-temporal history that 
    the model will learn to predict.
    """),
            seq_evolution_widget,
        ]
    )
    return


@app.cell(hide_code=True)
def zipfian_intro_md(mo):
    mo.md(r"""
    ### Why NCA Data Works: The Zipfian Connection

    We've seen how NCA grids become token streams. But why should a language model *benefit* from training on them? The answer lies in a statistical fingerprint shared by NCA data and natural language: the **Zipfian distribution**.

    In any natural language corpus, a few words (like "the", "is", "of") appear extremely often, while most words are rare. Plot word frequency on a log-log scale and you get a straight line known as **Zipf's law**. This heavy-tailed structure is a hallmark of meaningful, structured data.

    NCA token sequences follow the same law. A few patch patterns dominate (common local structures), while most are rare (unusual configurations). This isn't a coincidence; it's a consequence of the spatial correlations that NCA rules create.

    Use the slider below to change the NCA alphabet size and watch how the Zipfian slope shifts. Compare it against the reference curves from real language corpora.
    """)
    return


@app.cell(hide_code=True)
def zipfian_precompute(F, NCARule, NCA_H, NCA_W, torch):
    import collections as _collections_z


    def _tokenize_trajectory_patches_z(frames, n_states, patch_size=2):
        """Convert list of grids to patch tokens."""
        tokens = []
        for grid in frames:
            H = len(grid)
            W = len(grid[0]) if H > 0 else 0
            for pr in range(0, H, patch_size):
                for pc in range(0, W, patch_size):
                    val = 0
                    for dr in range(patch_size):
                        for dc in range(patch_size):
                            val = val * n_states + grid[pr + dr][pc + dc]
                    tokens.append(val)
        return tokens


    def _generate_zipfian_for_n(n_states_val, n_trajectories=80, n_steps=20):
        """Generate NCA trajectories and return (probs, n_unique)."""
        all_tokens = []
        for _ in range(n_trajectories):
            rule = NCARule(n_states=n_states_val)
            grid = torch.randint(0, n_states_val, (NCA_H, NCA_W))
            frames = [grid.tolist()]
            with torch.no_grad():
                for _ in range(n_steps - 1):
                    x = (
                        F.one_hot(grid, n_states_val)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    grid = rule(x).squeeze(0).argmax(0)
                    frames.append(grid.tolist())
            all_tokens.extend(_tokenize_trajectory_patches_z(frames, n_states_val))
        counter = _collections_z.Counter(all_tokens)
        freqs = sorted(counter.values(), reverse=True)
        total = sum(freqs)
        return [f / total for f in freqs]


    # Precompute for all slider values
    zipfian_cache = {}
    for _n_val in range(2, 16):
        zipfian_cache[_n_val] = _generate_zipfian_for_n(_n_val)
    return (zipfian_cache,)


@app.cell(hide_code=True)
def zipf_slider_cell(mo):
    zipf_n_slider = mo.ui.slider(
        start=2,
        stop=15,
        step=1,
        value=10,
        label="NCA Alphabet Size (n)",
        show_value=True,
    )
    zipf_n_slider
    return (zipf_n_slider,)


@app.cell(hide_code=True)
def zipfian_plot(alt, mo, pd, zipf_n_slider, zipfian_cache):
    _n_states_for_zipf = zipf_n_slider.value
    _nca_probs = zipfian_cache[_n_states_for_zipf]

    # Build dataframe with rank-frequency data
    _rows = []
    for rank, prob in enumerate(_nca_probs, 1):
        _rows.append(
            {
                "rank": rank,
                "probability": prob,
                "source": f"NCA (n={_n_states_for_zipf})",
            }
        )

    # Reference Zipfian curves
    _n_ref = 500
    for i in range(1, _n_ref + 1):
        _rows.append(
            {
                "rank": i,
                "probability": 0.08 / (i**1.0),
                "source": "Natural Language (Zipf a~1.0)",
            }
        )
        _rows.append(
            {
                "rank": i,
                "probability": 0.12 / (i**0.8),
                "source": "Code (Zipf a~0.8)",
            }
        )

    _zdf = pd.DataFrame(_rows)

    _zipf_chart = (
        alt.Chart(_zdf)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "rank:Q", scale=alt.Scale(type="log"), title="Token Rank (log)"
            ),
            y=alt.Y(
                "probability:Q",
                scale=alt.Scale(type="log"),
                title="Probability (log)",
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=[
                        f"NCA (n={_n_states_for_zipf})",
                        "Natural Language (Zipf a~1.0)",
                        "Code (Zipf a~0.8)",
                    ],
                    range=["#e15759", "#4e79a7", "#59a14f"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            strokeDash=alt.StrokeDash(
                "source:N",
                scale=alt.Scale(
                    domain=[
                        f"NCA (n={_n_states_for_zipf})",
                        "Natural Language (Zipf a~1.0)",
                        "Code (Zipf a~0.8)",
                    ],
                    range=[[0], [6, 4], [6, 4]],
                ),
                legend=None,
            ),
        )
        .properties(
            width=600,
            height=350,
            title="Log-Log Rank-Frequency: NCA tokens vs Natural Language",
        )
    )

    mo.vstack(
        [
            _zipf_chart,
            mo.callout(
                mo.md(
                    f"**NCA tokens (n={_n_states_for_zipf})** produce {len(_nca_probs):,} unique patch types from 80 trajectories. "
                    f"The heavy-tailed distribution closely matches the Zipfian shape of natural language. "
                    f"a few patterns dominate while most are rare. This shared statistical structure is why "
                    f"NCA pre-training transfers to language tasks."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### So what does the model actually learn?

    The training objective is identical to language modeling: predict the next token. But there is a crucial difference. In language, the model can exploit shallow shortcuts like word co-occurrence ("the cat sat on the ___"). In NCA sequences, there are no semantic shortcuts. Every sequence is generated by a **unique, never-before-seen rule**. The only way to predict the next token is to **figure out what that rule is**, purely from context.

    This leads to the central question: how does a Transformer go from seeing random-looking grids to accurately predicting the next one?
    """)
    return


@app.cell(hide_code=True)
def aha_moment_intro(mo):
    mo.md(r"""
    ### The "Aha!" Moment: In-Context Rule Inference

    How does a Transformer actually learn from these grids? It doesn't just memorize shapes. Instead, it performs **Implicit Bayesian Inference**.

    When the model sees the first grid, it is guessing. But once it sees the transition from the first grid to the second, it has enough evidence to "solve" the hidden neural rule ($\theta$). From that point on, it isn't guessing anymore; it is simulating the NCA in its internal layers.

    Let's watch this happen in real time. Below we generate a single NCA sequence and train a fresh TinyGPT on *just that one sequence*. At first the model's predictions are random noise. Then, after enough exposure, the loss drops sharply and the predicted grids snap into alignment with the real ones. That's the "Aha!" moment.
    """)
    return


@app.cell(hide_code=True)
def aha_train_button(mo):
    aha_train_button = mo.ui.run_button(
        label="Train on a Single NCA Sequence", kind="success"
    )
    aha_train_button
    return (aha_train_button,)


@app.cell(hide_code=True)
def aha_trainer(F, NCARule, TinyGPT, aha_train_button, alt, mo, pd, torch):
    mo.stop(not aha_train_button.value)

    import time as _time_aha

    # --- Config for the Aha demo: binary 8x8 grid ---
    _AHA_N = 4  # binary alphabet
    _AHA_H = 8
    _AHA_W = 8
    _AHA_GRID_TOKENS = _AHA_H * _AHA_W  # 64 tokens per frame
    _AHA_N_FRAMES = 10

    # 1. Generate a single NCA trajectory
    _aha_rule = NCARule(n_states=_AHA_N)
    _aha_grid = torch.randint(0, _AHA_N, (_AHA_H, _AHA_W))
    _aha_frames = [_aha_grid.clone()]
    with torch.no_grad():
        for _ in range(_AHA_N_FRAMES - 1):
            _x = F.one_hot(_aha_grid, _AHA_N).float().permute(2, 0, 1).unsqueeze(0)
            _aha_grid = _aha_rule(_x).squeeze(0).argmax(0)
            _aha_frames.append(_aha_grid.clone())

    # 2. Build training data: each sample is one frame (64 tokens) predicting the next frame
    #    Input: frame[t] flattened, Target: frame[t+1] flattened
    _aha_x_list = []
    _aha_y_list = []
    for _t in range(len(_aha_frames) - 1):
        _aha_x_list.append(_aha_frames[_t].reshape(-1))
        _aha_y_list.append(_aha_frames[_t + 1].reshape(-1))
    _aha_x = torch.stack(_aha_x_list)  # (N_FRAMES-1, 64)
    _aha_y = torch.stack(_aha_y_list)  # (N_FRAMES-1, 64)

    # 3. Fresh model sized for this task
    _aha_model = TinyGPT(vocab=_AHA_N, seq_len=_AHA_GRID_TOKENS)
    _aha_opt = torch.optim.AdamW(_aha_model.parameters(), lr=0.003)
    _aha_log = []

    # Colors for binary grid
    _AHA_COLORS = [
      "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
      "#edc948","#b07aa1","#ff9da7","#9c755f",
    ]


    def _render_grid_html(grid_2d, label, border_color="#ddd"):
        if hasattr(grid_2d, "tolist"):
            grid_2d = grid_2d.tolist()
        rows_html = ""
        for row in grid_2d:
            cells_html = "".join(
                f'<div style="width:18px;height:18px;background:{_AHA_COLORS[int(v) % len(_AHA_COLORS)]};display:inline-block;border:0.5px solid #e0e0e0;"></div>'
                for v in row
            )
            rows_html += f'<div style="line-height:0;">{cells_html}</div>'
        return mo.Html(
            f'<div style="text-align:center;">'
            f'<div style="font-size:11px;color:#666;margin-bottom:4px;font-weight:600;">{label}</div>'
            f'<div style="border:2px solid {border_color};display:inline-block;border-radius:4px;">{rows_html}</div>'
            f"</div>"
        )


    # Pick a frame pair to visualize
    _viz_t = min(3, len(_aha_frames) - 2)
    _viz_current = _aha_frames[_viz_t]
    _viz_true = _aha_frames[_viz_t + 1]

    # 4. Training loop
    _grids_display = mo.md("*Initializing...*")
    _step = 0
    _target_acc = 0.95
    _max_steps = 2000
    _extra_after_target = 80  # keep going a bit after hitting target
    _steps_above = 0

    while _step < _max_steps:
        _step += 1
        _aha_model.train()
        # Train on all frame pairs each step
        _logits = _aha_model(_aha_x)
        _loss = F.cross_entropy(_logits.reshape(-1, _AHA_N), _aha_y.reshape(-1))
        _aha_opt.zero_grad()
        _loss.backward()
        _aha_opt.step()

        if _step % 5 == 0:
            _aha_model.eval()
            with torch.no_grad():
                _eval_logits = _aha_model(_aha_x)
                _eval_acc = (
                    (_eval_logits.argmax(-1) == _aha_y).float().mean().item()
                )
                _eval_loss = F.cross_entropy(
                    _eval_logits.reshape(-1, _AHA_N), _aha_y.reshape(-1)
                ).item()

            _aha_log.append(
                {"step": _step, "loss": _eval_loss, "accuracy": _eval_acc}
            )

            # Check if we've hit target
            if _eval_acc >= _target_acc:
                _steps_above += 5
                if _steps_above >= _extra_after_target:
                    # We've shown enough post-Aha
                    pass

            # Predict the visualization frame
            if _step % 20 == 0 or _step == 5:
                with torch.no_grad():
                    _pred_logits = _aha_model(_viz_current.reshape(-1).unsqueeze(0))
                    _pred_grid = _pred_logits[0].argmax(-1).reshape(_AHA_H, _AHA_W)
                _match_pct = (_pred_grid == _viz_true).float().mean().item()
                _pred_border = "#59a14f" if _match_pct > 0.7 else "#e15759"

                _grids_display = mo.hstack(
                    [
                        _render_grid_html(_viz_current, "Input Frame", "#4e79a7"),
                        mo.Html(
                            '<div style="font-size:24px;color:#999;padding:0 10px;align-self:center;">&#8594;</div>'
                        ),
                        _render_grid_html(_viz_true, "True Next Frame", "#59a14f"),
                        mo.Html(
                            '<div style="font-size:16px;color:#999;padding:0 6px;align-self:center;">vs</div>'
                        ),
                        _render_grid_html(
                            _pred_grid,
                            f"Predicted ({_match_pct:.0%})",
                            _pred_border,
                        ),
                    ],
                    justify="center",
                    align="center",
                    gap="0.5rem",
                )

            # Build chart
            _log_df = pd.DataFrame(_aha_log)
            _loss_line = (
                alt.Chart(_log_df)
                .mark_line(color="steelblue", strokeWidth=2)
                .encode(
                    x=alt.X("step:Q", title="Step"),
                    y=alt.Y(
                        "loss:Q",
                        title="Loss",
                        axis=alt.Axis(
                            titleColor="steelblue", labelColor="steelblue"
                        ),
                    ),
                )
            )
            _acc_line = (
                alt.Chart(_log_df)
                .mark_line(color="darkorange", strokeWidth=2)
                .encode(
                    x=alt.X("step:Q"),
                    y=alt.Y(
                        "accuracy:Q",
                        title="Accuracy",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(
                            titleColor="darkorange", labelColor="darkorange"
                        ),
                    ),
                )
            )
            _chart = (
                alt.layer(_loss_line, _acc_line)
                .resolve_scale(y="independent")
                .properties(
                    width=560,
                    height=200,
                    title="Learning a Single NCA Rule (binary 8x8 grid)",
                )
            )

            mo.output.replace(
                mo.vstack(
                    [
                        _grids_display,
                        _chart,
                        mo.md(
                            f"**Step {_step}** | Loss: `{_eval_loss:.3f}` | Accuracy: `{_eval_acc:.1%}`"
                        ),
                    ]
                )
            )

            # Stop condition
            if _steps_above >= _extra_after_target:
                break

    # Final prediction
    _aha_model.eval()
    with torch.no_grad():
        _pred_logits = _aha_model(_viz_current.reshape(-1).unsqueeze(0))
        _pred_grid = _pred_logits[0].argmax(-1).reshape(_AHA_H, _AHA_W)
    _match_pct = (_pred_grid == _viz_true).float().mean().item()
    _pred_border = "#59a14f" if _match_pct > 0.7 else "#e15759"

    _grids_display = mo.hstack(
        [
            _render_grid_html(_viz_current, "Input Frame", "#4e79a7"),
            mo.Html(
                '<div style="font-size:24px;color:#999;padding:0 10px;align-self:center;">&#8594;</div>'
            ),
            _render_grid_html(_viz_true, "True Next Frame", "#59a14f"),
            mo.Html(
                '<div style="font-size:16px;color:#999;padding:0 6px;align-self:center;">vs</div>'
            ),
            _render_grid_html(
                _pred_grid, f"Predicted ({_match_pct:.0%})", _pred_border
            ),
        ],
        justify="center",
        align="center",
        gap="0.5rem",
    )

    mo.output.replace(
        mo.vstack(
            [
                _grids_display,
                _chart,
                mo.callout(
                    mo.md(
                        f"After **{_aha_log[-1]['step']} steps**, accuracy jumped to **{_aha_log[-1]['accuracy']:.0%}** "
                        f"and the predicted grid matches **{_match_pct:.0%}** of the true next state. "
                        f"The model deduced the hidden NCA rule purely from the sequence and can now simulate it. "
                        f"This is the in-context learning circuit that transfers to language."
                    ),
                    kind="success",
                ),
            ]
        )
    )
    return


@app.cell(hide_code=True)
def part2_conclusion_md(mo):
    mo.md(r"""
    ## From Mechanics to Results

    We've now seen the full pipeline: NCA rules generate structured dynamics, those dynamics get serialized into token sequences, and a Transformer trained on those sequences is forced to develop in-context learning circuits just to solve the prediction task.

    The question that remains is whether this actually translates to better performance on real language. Does a model pre-pre-trained on abstract grid patterns genuinely learn faster when it encounters English, Python, or mathematics?

    In Part 3, we turn to the paper's experimental results to find out.
    """)
    return


@app.cell(hide_code=True)
def part3_header_md(mo):
    mo.md(r"""
    # Part 3: What the Paper Found

    The authors tested this idea at scale. They pre-pre-trained a **1.6B parameter** Llama-based transformer on 164M NCA tokens, then continued with standard language pre-training on **OpenWebText**. They compared against three baselines:

    - **Scratch**: random initialization (no pre-pre-training)
    - **C4**: pre-pre-trained on 1.6B tokens of real English text
    - **Dyck**: pre-pre-trained on formal bracket languages (a structured but linguistic baseline)

    If NCA pre-pre-training only provided a trivial head start, we would expect the baselines to catch up within a few billion tokens. Instead, what they found was striking.
    """)
    return


@app.cell(hide_code=True)
def convergence_race_controls(alt, mo, pd):
    # Paper's approximate perplexity curves for OpenWebText (read from Figure 2)
    # Format: (tokens_billions, scratch_ppl, nca_ppl, c4_ppl, dyck_ppl)
    CONVERGENCE_DATA = [
        (0.5, 22.0, 20.5, 21.5, 21.8),
        (1.0, 19.0, 17.8, 18.7, 18.9),
        (1.5, 17.5, 16.4, 17.2, 17.3),
        (2.0, 16.5, 15.5, 16.3, 16.3),
        (3.0, 15.8, 14.8, 15.6, 15.6),
        (4.0, 15.3, 14.4, 15.1, 15.2),
        (5.0, 15.0, 14.1, 14.9, 14.9),
        (6.0, 14.8, 13.9, 14.7, 14.7),
        (7.0, 14.6, 13.7, 14.5, 14.5),
        (8.0, 14.5, 13.6, 14.4, 14.4),
        (9.0, 14.4, 13.5, 14.3, 14.3),
    ]

    _models = ["Scratch", "NCA", "C4", "Dyck"]
    _model_colors = {
        "Scratch": "#4e79a7",
        "NCA": "#e15759",
        "C4": "#59a14f",
        "Dyck": "#b07aa1",
    }

    _all_rows = []
    for tokens_b, scratch, nca, c4, dyck in CONVERGENCE_DATA:
        for model_name, ppl in zip(_models, [scratch, nca, c4, dyck]):
            _all_rows.append(
                {"tokens_B": tokens_b, "perplexity": ppl, "model": model_name}
            )

    _conv_df = pd.DataFrame(_all_rows)

    _conv_chart = (
        alt.Chart(_conv_df)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=30))
        .encode(
            x=alt.X("tokens_B:Q", title="Tokens (Billions)"),
            y=alt.Y(
                "perplexity:Q",
                title="Perplexity (lower is better)",
                scale=alt.Scale(domain=[13, 23]),
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=_models, range=[_model_colors[m] for m in _models]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=650,
            height=380,
            title="Pre-training Convergence on OpenWebText (1.6B model)",
        )
    )

    _final = CONVERGENCE_DATA[-1]
    _ppl_improvement = round((_final[1] - _final[2]) / _final[1] * 100, 1)

    mo.vstack(
        [
            _conv_chart,
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_final[1]:.1f}",
                        label="Scratch (final)",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{_final[2]:.1f}", label="NCA (final)", bordered=True
                    ),
                    mo.stat(
                        value=f"{_ppl_improvement}%",
                        label="Improvement",
                        bordered=True,
                    ),
                ],
                justify="center",
                gap="1rem",
            ),
            mo.callout(
                mo.md(
                    f"NCA pre-pre-training (orange) consistently outperforms all baselines throughout training. "
                    f"The gap with Scratch (blue) persists and even widens over time. "
                    f"Most remarkably, NCA used only **164M synthetic tokens** for pre-pre-training, "
                    f"while C4 (green) used **1.6B tokens of real text**, yet NCA still wins."
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def convergence_to_domain_bridge(mo):
    mo.md(r"""
    The perplexity advantage persists throughout training and even widens over time. But the paper didn't stop there. They asked a follow-up question: **does the same NCA configuration work equally well for every type of text?**

    It turns out the answer is no, and the reason ties back to what we learned about complexity in Part 2.
    """)
    return


@app.cell(hide_code=True)
def domain_match_intro_md(mo):
    mo.md(r"""
    ## Domain-Matched Complexity: One Size Does Not Fit All

    The paper discovered that **the optimal NCA complexity depends on the target domain**.

    - **CodeParrot** (code, gzip ~32%) benefits most from the **30-40%** NCA complexity band
    - **OpenWebText** (web text, gzip ~70%) benefits most from the **50%+** NCA complexity band
    - **OpenWebMath** (math, gzip ~58%) also favors **higher complexity** NCA rules, similar to web text

    The principle is straightforward: **match the synthetic data's complexity to your target domain's complexity**. Code is highly structured and repetitive, so simpler NCA patterns are the best warm-up. Natural language and math have richer, less predictable structure, so they benefit from more complex NCA dynamics.
    """)
    return


@app.cell(hide_code=True)
def domain_match_display(alt, mo, pd):
    # Perplexity improvement (%) by NCA band and target domain
    # Source: Paper Figure 6 (approximate), only OpenWebText and CodeParrot have per-band data
    _dm_rows = [
        {
            "NCA Band": "20-30%",
            "Domain": "CodeParrot (code)",
            "Improvement (%)": 2.0,
        },
        {
            "NCA Band": "30-40%",
            "Domain": "CodeParrot (code)",
            "Improvement (%)": 4.0,
        },
        {
            "NCA Band": "40-50%",
            "Domain": "CodeParrot (code)",
            "Improvement (%)": 3.0,
        },
        {"NCA Band": "50%+", "Domain": "CodeParrot (code)", "Improvement (%)": 2.0},
        {
            "NCA Band": "20-30%",
            "Domain": "OpenWebText (web)",
            "Improvement (%)": 1.0,
        },
        {
            "NCA Band": "30-40%",
            "Domain": "OpenWebText (web)",
            "Improvement (%)": 2.0,
        },
        {
            "NCA Band": "40-50%",
            "Domain": "OpenWebText (web)",
            "Improvement (%)": 3.0,
        },
        {"NCA Band": "50%+", "Domain": "OpenWebText (web)", "Improvement (%)": 5.0},
    ]
    _dm_df = pd.DataFrame(_dm_rows)

    _heatmap = (
        alt.Chart(_dm_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(
                "NCA Band:N",
                sort=["20-30%", "30-40%", "40-50%", "50%+"],
                title="NCA Complexity Band",
            ),
            y=alt.Y("Domain:N", title=None),
            color=alt.Color(
                "Improvement (%):Q",
                scale=alt.Scale(scheme="yellowgreenblue", domain=[0, 5]),
                legend=alt.Legend(title="Perplexity\nImprovement (%)"),
            ),
        )
    )

    _text = (
        alt.Chart(_dm_df)
        .mark_text(fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("NCA Band:N", sort=["20-30%", "30-40%", "40-50%", "50%+"]),
            y=alt.Y("Domain:N"),
            text=alt.Text("Improvement (%):Q", format=".1f"),
            color=alt.condition(
                alt.datum["Improvement (%)"] > 3.5,
                alt.value("white"),
                alt.value("#333"),
            ),
        )
    )

    _dm_chart = (_heatmap + _text).properties(
        width=400,
        height=150,
        title="Perplexity Improvement by NCA Band and Target Domain",
    )

    # Gzip reference markers
    _ref_rows = [
        {"Domain": "CodeParrot (code)", "gzip": 32},
        {"Domain": "OpenWebText (web)", "gzip": 70},
    ]
    _ref_df = pd.DataFrame(_ref_rows)

    _gzip_bars = (
        alt.Chart(_ref_df)
        .mark_bar(cornerRadiusEnd=4, height=20)
        .encode(
            x=alt.X(
                "gzip:Q",
                title="Domain's Own Gzip Complexity (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("Domain:N", title=None),
            color=alt.Color(
                "Domain:N",
                scale=alt.Scale(
                    domain=["CodeParrot (code)", "OpenWebText (web)"],
                    range=["#4e79a7", "#e15759"],
                ),
                legend=None,
            ),
        )
        .properties(
            width=400, height=80, title="Intrinsic Complexity of Each Domain"
        )
    )

    mo.vstack(
        [
            _dm_chart,
            mo.md(""),
            _gzip_bars,
            mo.callout(
                mo.md(
                    "The diagonal pattern is clear: **CodeParrot** (gzip 32%) peaks at the **30-40%** NCA band, "
                    "while **OpenWebText** (gzip 70%) peaks at the **50%+** band. "
                    "Each domain benefits most when the synthetic pre-training data matches its own complexity level."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def domain_to_weight_bridge(mo):
    mo.md(r"""
    So NCA pre-pre-training works, and its effectiveness can be tuned by matching complexity to the target domain. But *what is actually changing* inside the model? Which components absorb the benefit of pre-pre-training, and which ones are unaffected?

    To answer this, we can look directly at our own TinyGPT models from Part 1 and compare their weights.
    """)
    return


@app.cell(hide_code=True)
def weight_diff_intro_md(mo):
    mo.md(r"""
    ## What NCA Pre-training Does to the Weights

    The diagram below shows each component of TinyGPT as a block, colored by how much its weights diverged between the scratch-trained and NCA-pre-trained models. Hover over any block for the exact L2 difference. The bar chart on the right aggregates these differences by component type.
    """)
    return


@app.cell(hide_code=True)
def weight_diff_arch_widget(
    N_LAYERS,
    alt,
    anywidget,
    mo,
    nca_model,
    pd,
    scratch_model,
    traitlets,
):
    import numpy as _np_arch


    # Compute per-subcomponent diffs for the SVG widget
    def _compute_arch_diffs():
        diffs = {}
        for li in range(N_LAYERS):
            sb = scratch_model.blocks[li]
            nb = nca_model.blocks[li]
            s = sb.self_attn.in_proj_weight.detach().numpy()
            n = nb.self_attn.in_proj_weight.detach().numpy()
            diffs[f"attn_qkv_{li}"] = float(_np_arch.linalg.norm(n - s))
            s = sb.self_attn.out_proj.weight.detach().numpy()
            n = nb.self_attn.out_proj.weight.detach().numpy()
            diffs[f"attn_out_{li}"] = float(_np_arch.linalg.norm(n - s))
            s = sb.linear1.weight.detach().numpy()
            n = nb.linear1.weight.detach().numpy()
            diffs[f"mlp1_{li}"] = float(_np_arch.linalg.norm(n - s))
            s = sb.linear2.weight.detach().numpy()
            n = nb.linear2.weight.detach().numpy()
            diffs[f"mlp2_{li}"] = float(_np_arch.linalg.norm(n - s))
            s = sb.norm1.weight.detach().numpy()
            n = nb.norm1.weight.detach().numpy()
            diffs[f"ln1_{li}"] = float(_np_arch.linalg.norm(n - s))
            s = sb.norm2.weight.detach().numpy()
            n = nb.norm2.weight.detach().numpy()
            diffs[f"ln2_{li}"] = float(_np_arch.linalg.norm(n - s))
        s = scratch_model.ln_f.weight.detach().numpy()
        n = nca_model.ln_f.weight.detach().numpy()
        diffs["ln_f"] = float(_np_arch.linalg.norm(n - s))
        return diffs


    _arch_diffs = _compute_arch_diffs()


    # Compute aggregate per-component diffs for the bar chart
    def _compute_agg_diffs():
        components = []
        for li in range(N_LAYERS):
            sb = scratch_model.blocks[li]
            nb = nca_model.blocks[li]
            # Attention (QKV + Out combined)
            s_attn = sb.self_attn.in_proj_weight.detach().numpy()
            n_attn = nb.self_attn.in_proj_weight.detach().numpy()
            s_out = sb.self_attn.out_proj.weight.detach().numpy()
            n_out = nb.self_attn.out_proj.weight.detach().numpy()
            diff_attn = float(_np_arch.linalg.norm(n_attn - s_attn)) + float(
                _np_arch.linalg.norm(n_out - s_out)
            )
            components.append(
                {
                    "name": f"Attention L{li}",
                    "type": "attention",
                    "diff_norm": diff_attn,
                }
            )
            # MLP
            s_l1 = sb.linear1.weight.detach().numpy()
            n_l1 = nb.linear1.weight.detach().numpy()
            s_l2 = sb.linear2.weight.detach().numpy()
            n_l2 = nb.linear2.weight.detach().numpy()
            diff_mlp = float(_np_arch.linalg.norm(n_l1 - s_l1)) + float(
                _np_arch.linalg.norm(n_l2 - s_l2)
            )
            components.append(
                {"name": f"MLP L{li}", "type": "mlp", "diff_norm": diff_mlp}
            )
            # LayerNorm
            s_ln = sb.norm1.weight.detach().numpy()
            n_ln = nb.norm1.weight.detach().numpy()
            diff_ln = float(_np_arch.linalg.norm(n_ln - s_ln))
            components.append(
                {
                    "name": f"LayerNorm L{li}",
                    "type": "layernorm",
                    "diff_norm": diff_ln,
                }
            )
        return components


    _agg_diffs = _compute_agg_diffs()
    _wd_df = pd.DataFrame(_agg_diffs)

    _type_colors = {
        "attention": "#e15759",
        "mlp": "#4e79a8",
        "layernorm": "#76b7b2",
    }

    _diff_chart = (
        alt.Chart(_wd_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("diff_norm:Q", title="Weight Difference (L2 Norm)"),
            y=alt.Y("name:N", sort=None, title=None),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=list(_type_colors.keys()),
                    range=list(_type_colors.values()),
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["name:N", alt.Tooltip("diff_norm:Q", format=".2f")],
        )
        .properties(width=320, height=250, title="Weight Divergence (L2 Norm)")
    )

    _attn_avg = _wd_df[_wd_df["type"] == "attention"]["diff_norm"].mean()
    _mlp_avg = _wd_df[_wd_df["type"] == "mlp"]["diff_norm"].mean()
    _ratio = _attn_avg / _mlp_avg if _mlp_avg > 0 else float("inf")

    # SVG architecture widget
    _ARCH_ESM = """
    function render({ model, el }) {
      const diffs = model.get("diffs");
      const nLayers = model.get("n_layers");
      const maxDiff = Math.max(...Object.values(diffs));

      const W = 520, LH = 130, PAD = 15;
      const totalH = PAD + 20 + nLayers * LH + 70 + PAD;

      const root = document.createElement("div");
      root.style.cssText = "font-family: sans-serif; user-select: none;";

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("width", W);
      svg.setAttribute("height", totalH);
      svg.setAttribute("viewBox", `0 0 ${W} ${totalH}`);

      const tooltip = document.createElement("div");
      tooltip.style.cssText = "position:fixed;padding:8px 12px;background:#333;color:#fff;border-radius:6px;font-size:12px;pointer-events:none;opacity:0;transition:opacity 0.15s;z-index:999;font-family:monospace;";
      root.appendChild(tooltip);

      function intensity(key) { return Math.min(1, (diffs[key] || 0) / maxDiff); }

      function diffColor(key) {
        const t = intensity(key);
        if (t < 0.33) {
          const s = t / 0.33;
          return `rgb(255, 255, ${Math.round(255 - s * 100)})`;
        } else if (t < 0.66) {
          const s = (t - 0.33) / 0.33;
          return `rgb(255, ${Math.round(255 - s * 120)}, ${Math.round(155 - s * 100)})`;
        } else {
          const s = (t - 0.66) / 0.34;
          return `rgb(${Math.round(255 - s * 55)}, ${Math.round(135 - s * 80)}, ${Math.round(55 - s * 55)})`;
        }
      }

      function makeBlock(x, y, w, h, key, label, group) {
        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.style.cursor = "pointer";
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x); rect.setAttribute("y", y);
        rect.setAttribute("width", w); rect.setAttribute("height", h);
        rect.setAttribute("rx", 6); rect.setAttribute("fill", diffColor(key));
        rect.setAttribute("stroke", group === "attention" ? "#c44e52" : group === "mlp" ? "#4c78a8" : "#888");
        rect.setAttribute("stroke-width", 2);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x + w/2); text.setAttribute("y", y + h/2);
        text.setAttribute("dy", "0.35em"); text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "10px"); text.setAttribute("font-weight", "600");
        text.setAttribute("fill", "#333"); text.textContent = label;
        g.appendChild(rect); g.appendChild(text);
        g.addEventListener("mouseenter", (e) => {
          tooltip.innerHTML = `<b>${label}</b><br>L2 diff: ${(diffs[key]||0).toFixed(2)}<br>Intensity: ${(intensity(key)*100).toFixed(0)}%`;
          tooltip.style.opacity = 1; rect.setAttribute("stroke-width", 3);
        });
        g.addEventListener("mousemove", (e) => {
          tooltip.style.left = (e.clientX+12)+"px"; tooltip.style.top = (e.clientY-40)+"px";
        });
        g.addEventListener("mouseleave", () => {
          tooltip.style.opacity = 0; rect.setAttribute("stroke-width", 2);
        });
        svg.appendChild(g);
      }

      function arrow(x1,y1,x2,y2) {
        const l = document.createElementNS("http://www.w3.org/2000/svg","line");
        l.setAttribute("x1",x1);l.setAttribute("y1",y1);l.setAttribute("x2",x2);l.setAttribute("y2",y2);
        l.setAttribute("stroke","#bbb");l.setAttribute("stroke-width",1.5);
        l.setAttribute("marker-end","url(#ah)"); svg.appendChild(l);
      }
      function lbl(x,y,t,c) {
        const e=document.createElementNS("http://www.w3.org/2000/svg","text");
        e.setAttribute("x",x);e.setAttribute("y",y);e.setAttribute("font-size","10px");
        e.setAttribute("fill",c||"#999");e.setAttribute("font-weight","bold");e.textContent=t;
        svg.appendChild(e);
      }

      const defs=document.createElementNS("http://www.w3.org/2000/svg","defs");
      const mk=document.createElementNS("http://www.w3.org/2000/svg","marker");
      mk.setAttribute("id","ah");mk.setAttribute("markerWidth",8);mk.setAttribute("markerHeight",6);
      mk.setAttribute("refX",8);mk.setAttribute("refY",3);mk.setAttribute("orient","auto");
      const p=document.createElementNS("http://www.w3.org/2000/svg","path");
      p.setAttribute("d","M0,0 L8,3 L0,6 Z");p.setAttribute("fill","#bbb");
      mk.appendChild(p);defs.appendChild(mk);svg.appendChild(defs);

      const title=document.createElementNS("http://www.w3.org/2000/svg","text");
      title.setAttribute("x",W/2);title.setAttribute("y",16);title.setAttribute("text-anchor","middle");
      title.setAttribute("font-size","13px");title.setAttribute("font-weight","bold");title.setAttribute("fill","#444");
      title.textContent="TinyGPT Architecture (hover for details)";
      svg.appendChild(title);

      let cy=32; const cx=W/2; const bh=28;
      lbl(cx-15,cy,"Input","#aaa"); cy+=14; arrow(cx,cy,cx,cy+10); cy+=16;

      for(let li=0;li<nLayers;li++){
        const by=cy;
        const bg=document.createElementNS("http://www.w3.org/2000/svg","rect");
        bg.setAttribute("x",50);bg.setAttribute("y",by-4);bg.setAttribute("width",W-100);
        bg.setAttribute("height",LH-15);bg.setAttribute("rx",8);bg.setAttribute("fill","#f8f8f8");
        bg.setAttribute("stroke","#e0e0e0");bg.setAttribute("stroke-width",1);svg.appendChild(bg);
        lbl(58,by+10,"LAYER "+li,"#666");
        makeBlock(120,by+1,75,bh,"ln1_"+li,"LayerNorm","norm");
        makeBlock(210,by+1,110,bh,"attn_qkv_"+li,"Attn Q/K/V","attention");
        makeBlock(335,by+1,90,bh,"attn_out_"+li,"Attn Out","attention");
        arrow(cx,by+bh+5,cx,by+bh+14);
        makeBlock(120,by+bh+18,75,bh,"ln2_"+li,"LayerNorm","norm");
        makeBlock(210,by+bh+18,110,bh,"mlp1_"+li,"FFN Up","mlp");
        makeBlock(335,by+bh+18,90,bh,"mlp2_"+li,"FFN Down","mlp");
        cy+=LH;
        if(li<nLayers-1){arrow(cx,cy-10,cx,cy+2);cy+=6;}
      }
      cy+=8;arrow(cx,cy-12,cx,cy);
      makeBlock(cx-50,cy,100,bh,"ln_f","Final LN","norm");
      cy+=bh+10;arrow(cx,cy-6,cx,cy+4);lbl(cx-18,cy+16,"Output","#aaa");

      cy+=30;
      [["Low diff","rgb(255,255,255)"],["Medium","rgb(255,200,80)"],["High diff","rgb(200,55,0)"]].forEach((item,i)=>{
        const bx=80+i*155;
        const r=document.createElementNS("http://www.w3.org/2000/svg","rect");
        r.setAttribute("x",bx);r.setAttribute("y",cy);r.setAttribute("width",14);r.setAttribute("height",14);
        r.setAttribute("rx",3);r.setAttribute("fill",item[1]);r.setAttribute("stroke","#999");svg.appendChild(r);
        const t=document.createElementNS("http://www.w3.org/2000/svg","text");
        t.setAttribute("x",bx+20);t.setAttribute("y",cy+11);t.setAttribute("font-size","10px");
        t.setAttribute("fill","#666");t.textContent=item[0];svg.appendChild(t);
      });
      const ly2=cy+20;
      [["\\u25A0 Attention","#c44e52"],["\\u25A0 MLP","#4c78a8"],["\\u25A0 LayerNorm","#888"]].forEach((item,i)=>{
        const t=document.createElementNS("http://www.w3.org/2000/svg","text");
        t.setAttribute("x",80+i*155);t.setAttribute("y",ly2);t.setAttribute("font-size","10px");
        t.setAttribute("fill",item[1]);t.setAttribute("font-weight","bold");t.textContent=item[0];
        svg.appendChild(t);
      });

      root.appendChild(svg);el.appendChild(root);
    }
    export default { render };
    """


    class ArchDiffWidget(anywidget.AnyWidget):
        _esm = _ARCH_ESM
        diffs = traitlets.Dict({}).tag(sync=True)
        n_layers = traitlets.Int(N_LAYERS).tag(sync=True)


    arch_widget = ArchDiffWidget(diffs=_arch_diffs, n_layers=N_LAYERS)

    # Layout: arch diagram left, chart + stats right
    mo.vstack([
        mo.hstack(
            [
                arch_widget,
                mo.vstack(
                    [
                        _diff_chart,
                        mo.hstack(
                            [
                                mo.stat(
                                    value=f"{_attn_avg:.1f}",
                                    label="Avg Attention Diff",
                                    bordered=True,
                                ),
                                mo.stat(
                                    value=f"{_mlp_avg:.1f}",
                                    label="Avg MLP Diff",
                                    bordered=True,
                                ),
                                mo.stat(
                                    value=f"{_ratio:.1f}x",
                                    label="Attn/MLP Ratio",
                                    bordered=True,
                                ),
                            ],
                            justify="center",
                            gap="1rem",
                        ),
                    ],
                    gap="0.5rem",
                ),
            ],
            gap="1.5rem",
            align="start",
        ),
        mo.callout(
            mo.md(
                f"**Attention weights diverged {_ratio:.1f}x more** than MLP weights between the two training regimes. "
                f"Attention layers are where the universal computational primitives "
                f"(pattern matching, in-context learning circuits) develop during NCA pre-training, while "
                f"MLP layers remain closer to their initial state."
            ),
            kind="info",
        ),
    ])

    return


@app.cell(hide_code=True)
def weight_to_attention_bridge(mo):
    mo.md(r"""
    The architecture diagram confirms that attention weights diverge far more than MLP weights. But does this divergence actually *matter* for downstream performance? The paper tested this by selectively resetting parts of the NCA-trained model back to random weights, then measuring what breaks.

    We can run the same experiment on our TinyGPT models right now.
    """)
    return


@app.cell(hide_code=True)
def attention_transfer_intro_md(mo):
    mo.md(r"""
    ## The Attention Transfer Experiment

    Below, we take the fine-tuned NCA model and reset specific weight groups back to random initialization. If attention carries the transferable primitives, resetting it should devastate performance. If MLPs are domain-specific, resetting them should barely matter.
    """)
    return


@app.cell(hide_code=True)
def attention_transfer_experiment(
    PREDICT_DIGITS,
    alt,
    make_batch,
    mo,
    nca_model,
    nn,
    pd,
    torch,
):
    import copy as _copy


    def _selective_reset_test(base_model, n_samples=256):
        """Test model performance with selective weight resets."""
        results = {}

        base_model.eval()
        with torch.no_grad():
            xv, yv = make_batch(n_samples)
            acc_full = (
                (
                    base_model(xv)[:, PREDICT_DIGITS:, :].argmax(-1)
                    == yv[:, PREDICT_DIGITS:]
                )
                .float()
                .mean()
                .item()
            )
        results["Full NCA Model"] = acc_full

        attn_reset = _copy.deepcopy(base_model)
        with torch.no_grad():
            for block in attn_reset.blocks:
                attn = block.self_attn
                nn.init.xavier_uniform_(attn.in_proj_weight)
                nn.init.xavier_uniform_(attn.out_proj.weight)
                if attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)
                nn.init.zeros_(attn.out_proj.bias)
        attn_reset.eval()
        with torch.no_grad():
            acc_no_attn = (
                (
                    attn_reset(xv)[:, PREDICT_DIGITS:, :].argmax(-1)
                    == yv[:, PREDICT_DIGITS:]
                )
                .float()
                .mean()
                .item()
            )
        results["Reset Attention"] = acc_no_attn

        mlp_reset = _copy.deepcopy(base_model)
        with torch.no_grad():
            for block in mlp_reset.blocks:
                nn.init.xavier_uniform_(block.linear1.weight)
                nn.init.xavier_uniform_(block.linear2.weight)
                nn.init.zeros_(block.linear1.bias)
                nn.init.zeros_(block.linear2.bias)
        mlp_reset.eval()
        with torch.no_grad():
            acc_no_mlp = (
                (
                    mlp_reset(xv)[:, PREDICT_DIGITS:, :].argmax(-1)
                    == yv[:, PREDICT_DIGITS:]
                )
                .float()
                .mean()
                .item()
            )
        results["Reset MLP"] = acc_no_mlp

        both_reset = _copy.deepcopy(base_model)
        with torch.no_grad():
            for block in both_reset.blocks:
                attn = block.self_attn
                nn.init.xavier_uniform_(attn.in_proj_weight)
                nn.init.xavier_uniform_(attn.out_proj.weight)
                if attn.in_proj_bias is not None:
                    nn.init.zeros_(attn.in_proj_bias)
                nn.init.zeros_(attn.out_proj.bias)
                nn.init.xavier_uniform_(block.linear1.weight)
                nn.init.xavier_uniform_(block.linear2.weight)
                nn.init.zeros_(block.linear1.bias)
                nn.init.zeros_(block.linear2.bias)
        both_reset.eval()
        with torch.no_grad():
            acc_both = (
                (
                    both_reset(xv)[:, PREDICT_DIGITS:, :].argmax(-1)
                    == yv[:, PREDICT_DIGITS:]
                )
                .float()
                .mean()
                .item()
            )
        results["Reset Both"] = acc_both

        return results


    _nca_results = _selective_reset_test(nca_model)

    _config_colors = {
        "Full NCA Model": "#59a14f",
        "Reset Attention": "#e15759",
        "Reset MLP": "#4e79a7",
        "Reset Both": "#999999",
    }

    _at_df = pd.DataFrame(
        [{"config": k, "accuracy": v} for k, v in _nca_results.items()]
    )

    _at_chart = (
        alt.Chart(_at_df)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "accuracy:Q",
                title="Reversal Accuracy",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("config:N", sort=list(_nca_results.keys()), title=None),
            color=alt.Color(
                "config:N",
                scale=alt.Scale(
                    domain=list(_config_colors.keys()),
                    range=list(_config_colors.values()),
                ),
                legend=None,
            ),
        )
        .properties(
            width=500,
            height=200,
            title="Effect of Selective Weight Reset on NCA-Trained Model",
        )
    )

    _full_acc = _nca_results["Full NCA Model"]
    _attn_acc = _nca_results["Reset Attention"]
    _mlp_acc = _nca_results["Reset MLP"]
    _attn_drop = (_full_acc - _attn_acc) / _full_acc * 100 if _full_acc > 0 else 0
    _mlp_drop = (_full_acc - _mlp_acc) / _full_acc * 100 if _full_acc > 0 else 0

    mo.vstack(
        [
            _at_chart,
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_full_acc:.1%}", label="Full Model", bordered=True
                    ),
                    mo.stat(
                        value=f"{_attn_acc:.1%}",
                        label="Reset Attention",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{_mlp_acc:.1%}", label="Reset MLP", bordered=True
                    ),
                    mo.stat(
                        value=f"{_nca_results['Reset Both']:.1%}",
                        label="Reset Both",
                        bordered=True,
                    ),
                ],
                justify="center",
                gap="1rem",
            ),
            mo.callout(
                mo.md(
                    f"**Resetting attention** drops accuracy by **{_attn_drop:.0f}%**, destroying the learned "
                    f"computational primitives.\n\n"
                    f"**Resetting MLP** drops accuracy by only **{_mlp_drop:.0f}%**. The MLP stores "
                    f"task-specific patterns that are less critical to the model's core reasoning ability.\n\n"
                    f"This confirms the paper's finding: **attention layers are the universal carrier of transferable capabilities**."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def notebook_conclusion_md(mo):
    mo.md(r"""
    ## Putting It All Together

    We started with a simple question: *is natural language the only path to teaching a model to reason?*

    Through this notebook, we've seen that the answer is no. Abstract patterns from Neural Cellular Automata, containing no words, no grammar, and no meaning, can teach a Transformer the computational foundations it needs to learn language faster and more efficiently.

    **What we demonstrated:**

    1. **Pre-pre-training works.** Even our tiny TinyGPT model learns the reversal task faster after warming up on NCA patterns, with lower total energy cost.

    2. **NCA mechanics.** Neural cellular automata replace fixed rules with learned neural networks, creating diverse structured dynamics from simple local interactions.

    3. **Complexity is the key.** Gzip compressibility acts as a practical filter, selecting NCA rules in the "Goldilocks zone" between trivial repetition and chaotic noise. These rules produce token distributions that follow the same Zipfian power law as natural language.

    4. **Domain matching matters.** The optimal NCA complexity band varies by target domain: simpler for code, more complex for web text and math.

    5. **Attention carries the transfer.** The universal computational primitives learned during NCA pre-training live in the attention layers, not the MLPs. Resetting attention destroys the benefit; resetting MLPs barely matters.

    **The bigger picture:** If 164M synthetic NCA tokens can outperform 1.6B tokens of real language as a warm-up, perhaps the path to better AI isn't just *more* data but *better structured* data. The structural foundations of intelligence might be simpler than we thought.

    ---

    *Based on [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055) by Dan Lee, Seungwook Han, Akarsh Kumar, and Pulkit Agrawal (MIT / Improbable AI Lab).*
    """)
    return


if __name__ == "__main__":
    app.run()
