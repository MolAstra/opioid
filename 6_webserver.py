#!/usr/bin/env python3

from __future__ import annotations

import gradio as gr

from opioid.web import RFWebPredictor
from opioid.web.rf_web import MAX_CSV_ROWS


predictor = RFWebPredictor()


def predict_single_smiles(smiles: str):
    try:
        return predictor.predict_single(smiles)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def predict_csv_file(file_obj):
    if file_obj is None:
        raise gr.Error("Please upload a CSV file.")
    try:
        csv_path = file_obj if isinstance(file_obj, str) else file_obj.name
        return predictor.predict_csv(csv_path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Opioid RF Predictor") as demo:
        gr.Markdown(
            "\n".join(
                [
                    "# Opioid RF Predictor",
                    "RF antagonist classifier.",
                    "`pred_label`: `1 = antagonistic`, `0 = non-antagonistic`.",
                    f"CSV must include `smiles` or `SMILES` and contain at most {MAX_CSV_ROWS} rows.",
                ]
            )
        )

        with gr.Tabs():
            with gr.Tab("Single SMILES"):
                single_input = gr.Textbox(label="SMILES", lines=3, placeholder="Enter one SMILES")
                single_button = gr.Button("Predict", variant="primary")
                single_output = gr.Dataframe(
                    label="Prediction",
                    headers=["smiles", "antagonist_probability", "pred_label"],
                    interactive=False,
                )
                single_button.click(
                    fn=predict_single_smiles,
                    inputs=single_input,
                    outputs=single_output,
                )

            with gr.Tab("Batch CSV"):
                file_input = gr.File(label="Upload CSV", file_types=[".csv"], type="filepath")
                batch_button = gr.Button("Run Batch Prediction", variant="primary")
                batch_table = gr.Dataframe(label="Prediction Table", interactive=False)
                batch_file = gr.File(label="Download Predictions")
                batch_button.click(
                    fn=predict_csv_file,
                    inputs=file_input,
                    outputs=[batch_table, batch_file],
                )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
