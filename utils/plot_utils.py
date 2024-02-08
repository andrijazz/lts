import wandb
import torch
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "browser"


def plot_vis(id_sample, ood_sample):
    wandb.init(project="losh")

    fig = go.Figure()
    hist = torch.histogram(id_sample['penultimate_activation'], bins=200, range=[0, 10])
    amax = torch.amax(id_sample['penultimate_activation']).item()
    fig.add_trace(go.Scatter(x=hist.bin_edges[:-1],
                             y=hist.hist,
                             mode='lines',
                             name='ID',
                             line=dict(color='green')))
    fig.add_vline(x=amax, line_width=3, line_dash="dash", line_color="green", annotation_text="max ID")

    hist = torch.histogram(ood_sample['penultimate_activation'], bins=200, range=[0, 10])
    amax = torch.amax(ood_sample['penultimate_activation']).item()
    fig.add_trace(go.Scatter(x=hist.bin_edges[:-1],
                             y=hist.hist,
                             mode='lines',
                             name='OOD',
                             line=dict(color='red')))
    fig.add_vline(x=amax, line_width=3, line_dash="dash", line_color="red", annotation_text="max OOD")

    fig_logits = go.Figure()
    hist = torch.histogram(id_sample['logits'], bins=200, range=[-20, 20])
    fig_logits.add_trace(go.Scatter(x=hist.bin_edges[:-1],
                                    y=hist.hist,
                                    mode='lines',
                                    name='ID',
                                    line=dict(color='green')))

    hist = torch.histogram(ood_sample['logits'], bins=200, range=[-20, 20])
    fig_logits.add_trace(go.Scatter(x=hist.bin_edges[:-1],
                                    y=hist.hist,
                                    mode='lines',
                                    name='OOD',
                                    line=dict(color='red')))

    wandb.log({
        f"id_image": wandb.Image(id_sample['image'], caption=f"{id_sample['dataset']} image"),
        f"ood_image": wandb.Image(ood_sample['image'], caption=f"{ood_sample['dataset']} image"),
        f"penultimate_activation_distribution": fig,
        f"logits_distribution": fig_logits
    })
