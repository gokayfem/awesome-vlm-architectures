#!/usr/bin/env python3
"""Generate the chronological model index, timeline, and architecture catalog."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# The 58-entry catalog expansion carries Released metadata inside its README panels.
# These immutable migration IDs retain the primary-source audit for the 97 legacy
# panels; they are converted to a title-keyed lookup before any chronological sort.
# Values are (date, basis, primary source).
LEGACY_RELEASES_BY_MIGRATION_ID: dict[int, tuple[str, str, str]] = {
    59: ("2023-04-17", "arXiv v1", "https://arxiv.org/abs/2304.08485"),
    60: ("2023-10-05", "arXiv v1", "https://arxiv.org/abs/2310.03744"),
    61: ("2024-01-30", "official release", "https://llava-vl.github.io/blog/2024-01-30-llava-next/"),
    62: ("2024-05-14", "official release", "https://developers.googleblog.com/en/gemma-family-and-toolkit-expansion-io-2024/"),
    63: ("2024-12-04", "arXiv v1", "https://arxiv.org/abs/2412.03555"),
    64: ("2024-11-21", "arXiv v1", "https://arxiv.org/abs/2411.14402"),
    65: ("2024-12-13", "arXiv v1", "https://arxiv.org/abs/2412.10360"),
    66: ("2024-10-08", "arXiv v1", "https://arxiv.org/abs/2410.05993"),
    67: ("2024-06-17", "arXiv v1", "https://arxiv.org/abs/2406.11832"),
    68: ("2025-02-08", "official release", "https://huggingface.co/BAAI/EVE-7B-HD-v2.0"),
    69: ("2024-10-17", "arXiv v1", "https://arxiv.org/abs/2410.13848"),
    70: ("2024-11-15", "arXiv v1", "https://arxiv.org/abs/2411.10440"),
    71: ("2024-11-06", "official release", "https://github.com/microsoft/LLM2CLIP"),
    72: ("2024-12-10", "arXiv v1", "https://arxiv.org/abs/2412.07112"),
    73: ("2025-01-14", "arXiv v1", "https://arxiv.org/abs/2501.08313"),
    74: ("2024-09-17", "arXiv v1", "https://arxiv.org/abs/2409.11402"),
    75: ("2024-12-16", "arXiv v1", "https://arxiv.org/abs/2412.11475"),
    76: ("2024-09-11", "official release", "https://docs.mistral.ai/models/model-cards/pixtral-12b-24-09"),
    77: ("2025-01-07", "arXiv v1", "https://arxiv.org/abs/2501.04001"),
    78: ("2024-11-05", "official release", "https://github.com/bytedance/tarsier"),
    79: ("2025-01-20", "official release", "https://huggingface.co/bytedance-research/UI-TARS-7B-SFT"),
    80: ("2024-12-31", "arXiv v1", "https://arxiv.org/abs/2501.00574"),
    81: ("2025-01-21", "official release", "https://github.com/DAMO-NLP-SG/VideoLLaMA3"),
    82: ("2024-09-25", "official release", "https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/"),
    83: ("2024-11-26", "official release", "https://huggingface.co/blog/smolvlm"),
    84: ("2024-04-15", "official release", "https://huggingface.co/blog/idefics2"),
    85: ("2024-08-22", "arXiv v1", "https://arxiv.org/abs/2408.12637"),
    86: ("2024-01-29", "arXiv v1", "https://arxiv.org/abs/2401.16420"),
    87: ("2024-04-09", "arXiv v1", "https://arxiv.org/abs/2404.06512"),
    88: ("2024-07-03", "arXiv v1", "https://arxiv.org/abs/2407.03320"),
    89: ("2024-12-05", "official release", "https://internvl.github.io/blog/2024-12-05-InternVL-2.5/"),
    90: ("2024-03-08", "arXiv v1", "https://arxiv.org/abs/2403.05525"),
    91: ("2024-12-13", "arXiv v1", "https://arxiv.org/abs/2412.10302"),
    92: ("2024-05-02", "arXiv v1", "https://arxiv.org/abs/2405.01483"),
    93: ("2023-08-24", "arXiv v1", "https://arxiv.org/abs/2308.12966"),
    94: ("2024-08-29", "official release", "https://qwenlm.github.io/blog/qwen2-vl/"),
    95: ("2025-01-26", "official release", "https://qwenlm.github.io/blog/qwen2.5-vl/"),
    96: ("2024-01-20", "official release", "https://huggingface.co/vikhyatk/moondream1"),
    97: ("2024-04-19", "official release", "https://huggingface.co/vikhyatk/moondream-next"),
    98: ("2024-02-08", "arXiv v1", "https://arxiv.org/abs/2402.05935"),
    99: ("2022-01-28", "arXiv v1", "https://arxiv.org/abs/2201.12086"),
    100: ("2023-01-30", "arXiv v1", "https://arxiv.org/abs/2301.12597"),
    101: ("2024-05-06", "official release", "https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1"),
    102: ("2023-05-11", "arXiv v1", "https://arxiv.org/abs/2305.06500"),
    103: ("2023-02-27", "arXiv v1", "https://arxiv.org/abs/2302.14045"),
    104: ("2023-06-26", "arXiv v1", "https://arxiv.org/abs/2306.14824"),
    105: ("2024-05-24", "arXiv v1", "https://arxiv.org/abs/2405.15738"),
    106: ("2024-06-04", "arXiv v1", "https://arxiv.org/abs/2406.02539"),
    107: ("2024-06-27", "arXiv v1", "https://arxiv.org/abs/2406.19389"),
    108: ("2024-07-19", "arXiv v1", "https://arxiv.org/abs/2407.14177"),
    109: ("2024-07-22", "arXiv v1", "https://arxiv.org/abs/2407.15841"),
    110: ("2024-07-23", "arXiv v1", "https://arxiv.org/abs/2407.16198"),
    111: ("2024-07-24", "arXiv v1", "https://arxiv.org/abs/2407.17453"),
    112: ("2024-01-30", "official release", "https://huggingface.co/openbmb/MiniCPM-V"),
    113: ("2025-01-12", "official release", "https://huggingface.co/openbmb/MiniCPM-o-2_6"),
    114: ("2024-08-05", "official release", "https://llava-vl.github.io/blog/2024-08-05-llava-onevision/"),
    115: ("2024-08-09", "arXiv v1", "https://arxiv.org/abs/2408.05211"),
    116: ("2024-08-28", "arXiv v1", "https://arxiv.org/abs/2408.15998"),
    117: ("2025-01-10", "official release", "https://huggingface.co/nvidia/Eagle2-9B"),
    118: ("2023-11-10", "arXiv v1", "https://arxiv.org/abs/2311.06242"),
    119: ("2022-12-21", "arXiv v1", "https://arxiv.org/abs/2212.10773"),
    120: ("2024-01-30", "arXiv v1", "https://arxiv.org/abs/2401.17221"),
    121: ("2023-05-24", "arXiv v1", "https://arxiv.org/abs/2305.15023"),
    122: ("2023-11-28", "official release", "https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha"),
    123: ("2023-12-28", "arXiv v1", "https://arxiv.org/abs/2312.16862"),
    124: ("2023-11-06", "arXiv v1", "https://arxiv.org/abs/2311.03354"),
    125: ("2023-11-06", "arXiv v1", "https://arxiv.org/abs/2311.03356"),
    126: ("2024-01-01", "arXiv v1", "https://arxiv.org/abs/2401.00849"),
    127: ("2024-01-05", "official release", "https://huggingface.co/fireworks-ai/FireLLaVA-13b"),
    128: ("2023-11-09", "arXiv v1", "https://arxiv.org/abs/2311.05348"),
    129: ("2024-01-29", "arXiv v1", "https://arxiv.org/abs/2401.15947"),
    130: ("2023-08-19", "arXiv v1", "https://arxiv.org/abs/2308.09936"),
    131: ("2023-12-28", "arXiv v1", "https://arxiv.org/abs/2312.16886"),
    132: ("2021-06-25", "arXiv v1", "https://arxiv.org/abs/2106.13884"),
    133: ("2022-04-28", "official release", "https://deepmind.google/blog/tackling-multiple-tasks-with-a-single-visual-language-model/"),
    134: ("2023-03-14", "official release", "https://doi.org/10.5281/zenodo.7733589"),
    135: ("2023-08-22", "official release", "https://huggingface.co/blog/idefics"),
    136: ("2022-09-14", "arXiv v1", "https://arxiv.org/abs/2209.06794"),
    137: ("2023-10-13", "arXiv v1", "https://arxiv.org/abs/2310.09199"),
    138: ("2023-03-06", "arXiv v1", "https://arxiv.org/abs/2303.03378"),
    139: ("2023-04-16", "official release", "https://github.com/Vision-CAIR/MiniGPT-4/commit/f1a33af2274217f6d2b0bd60639a7e0e965392f0"),
    140: ("2023-10-13", "official release", "https://github.com/Vision-CAIR/MiniGPT-4"),
    141: ("2023-11-09", "arXiv v1", "https://arxiv.org/abs/2311.05437"),
    142: ("2023-10-12", "official release", "https://huggingface.co/SkunkworksAI/BakLLaVA-1"),
    143: ("2023-10-05", "official release", "https://github.com/THUDM/CogVLM"),
    144: ("2024-05-20", "official release", "https://github.com/THUDM/CogVLM2"),
    145: ("2023-10-11", "arXiv v1", "https://arxiv.org/abs/2310.07704"),
    146: ("2023-10-17", "official release", "https://www.adept.ai/blog/fuyu-8b"),
    147: ("2023-11-07", "arXiv v1", "https://arxiv.org/abs/2311.04219"),
    148: ("2023-11-13", "arXiv v1", "https://arxiv.org/abs/2311.07575"),
    149: ("2021-01-05", "official release", "https://openai.com/index/clip/"),
    150: ("2023-09-28", "arXiv v1", "https://arxiv.org/abs/2309.16671"),
    151: ("2023-12-06", "arXiv v1", "https://arxiv.org/abs/2312.03818"),
    152: ("2021-12-07", "arXiv v1", "https://arxiv.org/abs/2112.03857"),
    153: ("2023-05-09", "arXiv v1", "https://arxiv.org/abs/2305.05665"),
    154: ("2023-03-27", "arXiv v1", "https://arxiv.org/abs/2303.15343"),
    155: ("2020-10-22", "arXiv v1", "https://arxiv.org/abs/2010.11929"),
}

LEGACY_LABELS_BY_MIGRATION_ID = [
    "LLaVA", "LLaVA 1.5", "LLaVA 1.6", "PaliGemma", "PaliGemma 2", "AIMv2",
    "Apollo", "ARIA", "EVE", "EVEv2", "Janus and Janus-Pro", "LLaVA-CoT",
    "LLM2CLIP", "Maya", "MiniMax-01", "NVLM", "OmniVLM", "Pixtral 12B", "Sa2VA",
    "Tarsier2", "UI-TARS", "VideoChat-Flash", "VideoLLaMA 3", "Llama 3.2-Vision",
    "SmolVLM", "Idefics2", "Idefics3-8B", "InternLM-XComposer2",
    "InternLM-XComposer2-4KHD", "InternLM-XComposer-2.5", "InternVL 2.5", "DeepSeek-VL",
    "DeepSeek-VL2", "MANTIS", "Qwen-VL", "Qwen2-VL", "Qwen2.5-VL",
    "moondream1 and moondream2", "Moondream-next", "SPHINX-X", "BLIP", "BLIP-2",
    "xGen-MM (BLIP-3)", "InstructBLIP", "KOSMOS-1", "KOSMOS-2", "ConvLLaVA",
    "Parrot", "OMG-LLaVA", "EVLM", "SlowFast-LLaVA", "INF-LLaVA", "VILA²",
    "MiniCPM-V", "MiniCPM-o-2.6", "LLaVA-OneVision", "VITA", "EAGLE", "Eagle 2",
    "Florence-2", "MULTIINSTRUCT", "MouSi", "LaVIN", "Nous-Hermes-2-Vision - Mistral 7B",
    "TinyGPT-V", "CoVLM", "GLaMM", "COSMO", "FireLLaVA", "u-LLaVA", "MoE-LLaVA",
    "BLIVA", "MobileVLM", "FROZEN", "Flamingo", "OpenFlamingo", "IDEFICS", "PaLI",
    "PaLI-3 Vision Language Models", "PaLM-E", "MiniGPT-4", "MiniGPT-v2", "LLaVA-Plus",
    "BakLLaVA", "CogVLM", "CogVLM2", "Ferret", "Fuyu-8B", "OtterHD", "SPHINX", "CLIP",
    "MetaCLIP", "Alpha-CLIP", "GLIP", "ImageBind", "SigLIP", "ViT",
]

AUDITED_RELEASES_BY_LABEL = {
    label: LEGACY_RELEASES_BY_MIGRATION_ID[index]
    for index, label in zip(
        range(59, 156), LEGACY_LABELS_BY_MIGRATION_ID, strict=True
    )
}

# Preserve the hand-edited concise descriptions from the original timeline.
CONTRIBUTION_OVERRIDES = {
    "MODUS": "Decoder-only any-to-any modeling without modality-specific heads or losses",
    "Argus-Unified": "Hybrid continuous and discrete visual tokens for economical understanding and generation",
    "Kimi K3": "Kimi Delta Attention, Attention Residuals, and extremely sparse LatentMoE routing",
    "Mage-VL": "Codec-native selective video tokenization with a proactive event gate",
    "Inkling": "Relative-position million-context multimodal MoE trained from scratch",
    "Hy-Embodied-VLM": "Action-centric sparse-MoE reasoning for physical-world agents",
    "MonkeyOCRv2": "Joint image-to-text and pixel-reconstruction pretraining for document vision",
    "MiniMax M3": "Native multimodality with block-sparse grouped-query attention at million-token context",
    "InternVideo3": "Token-preserving latent KV compression and closed-loop video reasoning",
    "Keye-VL 2.0": "DeepSeek Sparse Attention adapted to GQA-based long-video multimodality",
    "Zamba2-VL": "Hybrid Mamba-2 and shared-attention blocks for efficient VLM inference",
    "Cosmos 3": "Coupled autoregressive reasoner and diffusion generator for physical AI",
    "Lance": "Shared-sequence understanding, generation, and editing with modality experts",
    "ZAYA1-VL": "Vision-conditional LoRA and compressed convolutional attention in an open-data MoE",
    "Falcon Perception": "Early fusion with hybrid attention and continuous mask heads",
    "GLM-5V-Turbo": "Perception integrated into reasoning, planning, tools, and execution",
    "PLaMo 2.1-VL": "Compact Japanese VQA and grounding for edge deployment",
    "EXAONE 4.5": "Native multimodal pretraining with document-focused data and 256K context",
    "BidirLM and BidirLM-Omni": "Converting causal decoders into bidirectional multimodal encoders",
    "Gemma 4": "Dense and MoE native multimodality, including an encoder-free 12B design",
    "Penguin-VL": "Text-LLM-initialized vision encoder and priority-aware token compression",
    "Phi-4-Reasoning-Vision": "Mid-fusion compact VLM with explicit reasoning and direct-answer modes",
    "V-SONAR and V-LCM": "Vision-language alignment and prediction in multilingual concept space",
    "Qwen3.5": "Native early fusion with hybrid linear/full attention and sparse MoE variants",
    "Youtu-VL": "Unified autoregressive visual tokens that emit dense vision outputs without task heads",
    "Kimi K2.5 and K2.6": "Trillion-parameter native multimodal MoE for agents and computer use",
    "Step3-VL-10B": "Language-aligned perception encoder with 16-fold visual-token compression",
    "ERNIE 5.0": "One autoregressive sparse MoE for text, images, video, audio, and generation",
    "DeepSeek-OCR": "DeepEncoder compresses high-resolution documents into very short visual contexts",
    "PaddleOCR-VL": "NaViT-style dynamic resolution with a compact ERNIE decoder for document parsing",
    "Qwen3-VL": "DeepStack multi-level ViT fusion and explicit video timestamp alignment",
    "Step3": "Model-system co-design for communication-efficient sparse-MoE multimodality",
    "GLM-4.1V-Thinking": "Curriculum-sampled reinforcement learning for multimodal reasoning",
    "ERNIE 4.5-VL": "Heterogeneous shared and modality-specific experts with isolated routing",
    "MiMo-VL": "Four-stage multimodal pretraining followed by mixed on-policy RL",
    "BAGEL": "Mixture-of-Transformer-Experts for understanding and generation",
    "Seed1.5-VL": "Compact vision encoder with a 20B-active MoE for reasoning and agents",
    "InternVL3 and InternVL3.5": "Native multimodal pretraining, later extended with adaptive resolution and cascade RL",
    "Kimi-VL": "MoonViT native-resolution packing with a sparse MoE decoder",
    "Llama 4 Scout and Maverick": "Early-fusion native multimodality in sparse-MoE Scout and Maverick models",
    "Qwen2.5-Omni": "Streaming Thinker-Talker architecture for multimodal input and speech output",
    "Gemma 3": "Efficient local/global attention with long-context image understanding",
    "Aya Vision": "Cross-modal model merging for multilingual multimodality without language forgetting",
    "Phi-4-multimodal": "Mixture-of-LoRAs for text, vision, and speech",
    "SigLIP 2": "Multilingual, localization-aware, native-aspect-ratio vision-language encoding",
    "ShowUI": "UI-guided visual-token selection and interleaved action histories",
    "Janus and Janus-Pro": "Decoupled visual encoders for understanding and generation with one transformer",
    "Emu3": "One next-token objective over discrete text, image, and video tokens",
    "Molmo and PixMo": "Open data pipeline with human captions and grounded pointing supervision",
    "VILA-U": "Shared discrete visual tokens for autoregressive understanding and generation",
    "Show-o": "Autoregressive language and discrete-diffusion image generation in one transformer",
    "Transfusion": "Autoregressive text and continuous image diffusion in one transformer",
    "mPLUG-Owl3": "Hyper-attention for long image sequences and video",
    "Cambrian-1": "Spatial Vision Aggregator and systematic multi-encoder study",
    "Ovis": "Learnable visual vocabulary for structural visual-text embedding alignment",
    "Phi-3-Vision and Phi-3.5-Vision": "Compact dynamic-resolution VLM with 128K context",
    "Chameleon": "Mixed-modal early fusion over a shared token sequence",
    "MM1": "Controlled study of encoders, connectors, token counts, and data mixtures",
    "AnyGPT": "Any-to-any autoregression over discrete text, image, speech, and music tokens",
    "moondream1 and moondream2": "Compact SigLIP–Phi VLMs optimized for efficient edge inference",
    "Idefics2": "Open 8B VLM with native-resolution inputs and strong OCR and document understanding",
    "FireLLaVA": "LLaVA derivative trained rapidly on a curated multimodal instruction mixture",
    "IDEFICS": "Open Flamingo-style model for interleaved image-text generation",
    "BakLLaVA": "Mistral-based LLaVA variant with a CLIP vision encoder and projection adapter",
    "Nous-Hermes-2-Vision - Mistral 7B": "SigLIP-equipped Mistral VLM with OCR and function-calling data",
}


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip().replace("|", "\\|")


def parse_sections(text: str) -> list[dict[str, object]]:
    start = text.index("## Architectures")
    end = text.index("## Important References", start)
    body = text[start:end]
    matches = list(re.finditer(r"(?m)^### \*\*(.+?)\*\*\s*$", body))
    sections: list[dict[str, object]] = []
    for position, match in enumerate(matches):
        section_end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
        raw = body[match.start() : section_end]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw)]
        contribution = next(
            (
                part for part in paragraphs[1:]
                if part and not part.startswith(("[!", "<", "<!--", "**Released:**"))
            ),
            None,
        )
        if contribution is None:
            raise ValueError(f"entry {position + 1} has no architecture summary")
        heading = match.group(1)
        label = heading.split(":", 1)[0]
        if label in CONTRIBUTION_OVERRIDES:
            contribution = CONTRIBUTION_OVERRIDES[label]
        elif ":" in heading:
            contribution = heading.split(":", 1)[1].strip()
        released = re.search(r"(?m)^\*\*Released:\*\* (\d{4}-\d{2}-\d{2})\s*$", raw)
        sections.append({
            "index": position + 1,
            "heading": heading,
            "label": label,
            "contribution": clean_inline(contribution),
            "released": released.group(1) if released else None,
            "start": start + match.start(),
            "end": start + section_end,
            "raw": raw,
        })
    return sections


def release_date(section: dict[str, object]) -> str:
    label = str(section["label"])
    released = section["released"]
    audited = AUDITED_RELEASES_BY_LABEL.get(label)
    if released and audited and released != audited[0]:
        raise ValueError(f"{label} has {released}, expected audited date {audited[0]}")
    if released:
        return str(released)
    if audited:
        return audited[0]
    raise ValueError(
        f"{label} has no release date; add **Released:** YYYY-MM-DD to its architecture panel"
    )


def chronological_sections(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return newest-first sections while preserving same-day editorial order."""
    return sorted(sections, key=release_date, reverse=True)


def github_anchor(heading: str) -> str:
    """Mirror GitHub's heading anchors for the characters used in this catalog."""
    cleaned = "".join(
        character
        for character in heading.lower()
        if character.isalnum() or character in {" ", "-", "_"}
    )
    return cleaned.replace(" ", "-")


def models_block(sections: list[dict[str, object]]) -> str:
    years: dict[str, list[str]] = {}
    for section in sections:
        year = release_date(section)[:4]
        label = str(section["label"])
        anchor = github_anchor(str(section["heading"]))
        years.setdefault(year, []).append(f"[{label}](#{anchor})")
    lines = [
        "## Models",
        "",
        "All architecture panels are ordered by release date, newest first. Models released on the same day retain editorial catalog order.",
        "",
        "<details>",
        f"<summary>🧭 <i>Chronological Model Index ({len(sections)} architectures, newest first)</i></summary>",
        "",
    ]
    for year, links in years.items():
        lines.extend([f"**{year}:** " + " | ".join(links), ""])
    lines.extend(["</details>", ""])
    return "\n".join(lines) + "\n"


def timeline_block(sections: list[dict[str, object]]) -> str:
    rows = []
    for section in sections:
        value = release_date(section)
        date.fromisoformat(value)
        rows.append((value, int(section["index"]), str(section["label"]), str(section["contribution"])))
    lines = [
        "## Release Timeline",
        "",
        "Dates use the first documented official model release; when none is available, they use the paper's arXiv v1 submission or first technical report. Family point releases are folded into their first architecture release, and same-day entries retain catalog order.",
        "",
        "<!--lint disable table-pipe-alignment table-cell-padding-->",
        "",
        "<details>",
        f"<summary>🗓️ <i>Release Timeline ({len(rows)} architectures, newest first)</i></summary>",
        "",
        "| Date | Architecture | Distinctive contribution |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {released} | {label} | {contribution} |" for released, _, label, contribution in rows)
    lines.extend([
        "",
        "</details>",
        "",
        "<!--lint enable table-pipe-alignment table-cell-padding-->",
        "",
        "",
    ])
    return "\n".join(lines)


def architectures_block(sections: list[dict[str, object]]) -> str:
    panels = "\n\n".join(str(section["raw"]).strip() for section in sections)
    return f"## Architectures\n\n{panels}\n\n"


def validate(text: str) -> None:
    sections = parse_sections(text)
    labels = [str(section["label"]) for section in sections]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate architecture labels: {duplicates}")
    dates = [release_date(section) for section in sections]
    if dates != sorted(dates, reverse=True):
        raise ValueError("architecture panels are not newest-first")

    models_start = text.index("## Models")
    models_end = text.index("## Release Timeline", models_start)
    model_links = re.findall(r"\[[^]]+]\(#([^)]+)\)", text[models_start:models_end])
    expected_links = [github_anchor(str(section["heading"])) for section in sections]
    if len(expected_links) != len(set(expected_links)):
        raise ValueError("duplicate GitHub heading anchors in architecture panels")
    if model_links != expected_links:
        raise ValueError("model index and architecture panels are not in identical order")

    start = text.index("## Release Timeline")
    end = text.index("## Architectures", start)
    timeline = text[start:end]
    rows = re.findall(r"(?m)^\| \d{4}-\d{2}-\d{2} \|", timeline)
    if len(rows) != len(sections):
        raise ValueError(f"expected {len(sections)} timeline rows, found {len(rows)}")
    table_records = re.findall(
        r"(?m)^\| (\d{4}-\d{2}-\d{2}) \| (.*?) \| .* \|$",
        timeline,
    )
    table_dates = [record[0] for record in table_records]
    table_labels = [record[1] for record in table_records]
    if table_dates != sorted(table_dates, reverse=True):
        raise ValueError("release timeline is not newest-first")
    if table_labels != labels:
        missing = sorted(set(labels) - set(table_labels))
        extra = sorted(set(table_labels) - set(labels))
        raise ValueError(
            "timeline/architecture order mismatch; "
            f"missing={missing}, extra={extra}"
        )
    expected_summary = (
        f"<summary>🗓️ <i>Release Timeline ({len(sections)} architectures, newest first)"
        "</i></summary>\n\n| Date"
    )
    if expected_summary not in timeline:
        raise ValueError("release timeline disclosure structure is invalid")


def build(original: str) -> str:
    sections = chronological_sections(parse_sections(original))
    models_start = original.index("## Models")
    references_start = original.index("## Important References")
    result = (
        original[:models_start]
        + models_block(sections)
        + timeline_block(sections)
        + architectures_block(sections)
        + original[references_start:]
    )
    validate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if README is not generated consistently")
    args = parser.parse_args()
    raw = README.read_bytes().decode("utf-8")
    original = raw.replace("\r\n", "\n")
    generated = build(original)
    count = len(parse_sections(generated))
    if args.check:
        if generated != original:
            raise SystemExit("README release timeline is out of date")
        print(f"release timeline is current: {count}/{count} architecture sections")
        return 0
    README.write_bytes(generated.encode("utf-8"))
    print(f"updated release timeline for {count} architecture sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
