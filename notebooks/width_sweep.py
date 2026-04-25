import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return Path, mo, project_root


@app.cell
def _(mo):
    mo.md(
        """
        # Scaling Ensembles: Width Sweep

        This app runs the first experiment for the paper idea: train several
        independently initialized networks per width, then measure how similar
        their functions are on a fixed evaluation set.
        """
    )
    return


@app.cell
def _(mo, project_root):
    config_path = mo.ui.text(
        value=str(project_root / "experiments" / "mnist_width_sweep.yaml"),
        label="Experiment config",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Run sweep")
    mo.vstack([config_path, run_button])
    return config_path, run_button


@app.cell
def _(config_path, mo):
    from scaling_ensembles.config import load_config

    config = load_config(config_path.value)
    mo.md(
        f"""
        **Experiment:** `{config.name}`

        **Widths:** `{list(config.model.widths)}`

        **Seeds:** `{list(config.training.seeds)}`

        **Output:** `{config.output_dir}`
        """
    )
    return config


@app.cell
def _(config_path, mo, run_button):
    outputs = None
    if run_button.value:
        from scaling_ensembles.sweep import run_sweep

        outputs = run_sweep(config_path.value)
        mo.md(
            "\n".join(
                [
                    "Sweep complete.",
                    *[f"- `{name}`: `{path}`" for name, path in outputs.items()],
                ]
            )
        )
    else:
        mo.md("Press **Run sweep** to train models and compute similarity metrics.")
    return outputs


@app.cell
def _(Path, config, mo, outputs, project_root):
    from scaling_ensembles.plots import plot_loss_barriers, plot_similarity_vs_params

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    pairwise_csv = output_dir / "pairwise_similarity.csv"
    interpolation_csv = output_dir / "interpolation_barriers.csv"

    figs = []
    if pairwise_csv.exists():
        figs.append(plot_similarity_vs_params(pairwise_csv))
    if interpolation_csv.exists():
        figs.append(plot_loss_barriers(interpolation_csv))

    if figs:
        mo.vstack(figs)
    elif outputs is None:
        mo.md("No result CSVs found yet.")
    else:
        mo.md("Sweep ran, but result CSVs were not found at the expected paths.")
    return


if __name__ == "__main__":
    app.run()
