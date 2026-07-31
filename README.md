# 👁️‍🗨️ Awesome VLM Architectures [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<!--lint disable double-link-->
![VLM](https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/5c9ee091-1f37-4d92-8398-a7d4e006c014)

**Vision-Language Models (VLMs)** feature a multimodal architecture that processes image and text data simultaneously. They can perform **Visual Question Answering (VQA)**, **image captioning** and **Text-To-Image search** kind of tasks. VLMs utilize techniques like multimodal fusing with cross-attention, masked-language modeling, and image-text matching to relate visual semantics to textual representations. This repository contains information on famous Vision Language Models (VLMs), including details about their architectures, training procedures, and the datasets used for training. **Click to expand for further details for every architecture**
- 📙 <a href="https://github.com/gokayfem/ComfyUI_VLM_nodes">Visit my other repo to try Vision Language Models on ComfyUI</a>

*Last reviewed: July 31, 2026.*

## Contents

- [Citation](#citation)
- [Models](#models)
- [Release Timeline](#release-timeline)
- [Architectures](#architectures)
- [Important References](#important-references)

## Citation

If this repository is useful in your work, you may cite it below. Please also cite the original paper for claims about any individual model; this catalog is a guide to the literature, not a substitute for it.

<details>
<summary>📚 <i>BibTeX</i></summary>

```bibtex
@misc{gokayfem2024awesomevlmarchitectures,
  author       = {{gokayfem}},
  title        = {Awesome VLM Architectures},
  year         = {2024},
  howpublished = {\url{https://github.com/gokayfem/awesome-vlm-architectures}},
  note         = {GitHub repository}
}
```

</details>

## Models

**2026 additions:** [MODUS](#modus-decoder-only-any-to-any-multimodal-modeling) | [Argus-Unified](#argus-unified-economical-understanding-and-generation) | [Kimi K3](#kimi-k3-kimi-delta-attention-at-trillion-parameter-scale) | [Mage-VL](#mage-vl-codec-native-streaming-multimodality) | [Inkling](#inkling-relative-position-multimodal-mixture-of-experts) | [Hy-Embodied-VLM](#hy-embodied-vlm-sparse-moe-reasoning-for-physical-agents) | [MonkeyOCRv2](#monkeyocrv2-document-native-visual-text-pretraining) | [MiniMax M3](#minimax-m3-native-multimodality-with-sparse-long-context-attention) | [InternVideo3](#internvideo3-multimodal-contextual-reasoning-for-video-agents) | [Keye-VL 2.0](#keye-vl-20-sparse-attention-for-long-video-agents) | [Zamba2-VL](#zamba2-vl-hybrid-state-space-vision-language-modeling) | [Cosmos 3](#cosmos-3-omnimodal-world-modeling-with-mixture-of-transformers) | [Lance](#lance-unified-image-and-video-understanding-generation-and-editing) | [ZAYA1-VL](#zaya1-vl-vision-specialized-compressed-convolutional-attention) | [Falcon Perception](#falcon-perception-early-fusion-grounding-segmentation-and-ocr) | [GLM-5V-Turbo](#glm-5v-turbo-native-multimodal-agency) | [PLaMo 2.1-VL](#plamo-21-vl-lightweight-japanese-vision-language-modeling) | [EXAONE 4.5](#exaone-45-native-multimodal-pretraining-for-documents) | [BidirLM / BidirLM-Omni](#bidirlm-and-bidirlm-omni-causal-decoders-as-multimodal-encoders) | [Gemma 4](#gemma-4-open-weight-native-multimodal-models) | [Penguin-VL](#penguin-vl-efficient-vlms-with-llm-based-vision-encoders) | [Phi-4-Reasoning-Vision](#phi-4-reasoning-vision-compact-multimodal-reasoning) | [V-SONAR / V-LCM](#v-sonar-and-v-lcm-vision-language-modeling-in-concept-space) | [Qwen3.5](#qwen35-native-multimodal-hybrid-attention-models) | [Youtu-VL](#youtu-vl-unified-autoregressive-supervision-for-dense-vision) | [Kimi K2.5 / K2.6](#kimi-k25-and-k26-native-multimodal-agentic-moe) | [Step3-VL-10B](#step3-vl-10b-language-aligned-perception-with-16-token-compression)

**2025 additions:** [ERNIE 5.0](#ernie-50-unified-autoregressive-omnimodal-mixture-of-experts) | [DeepSeek-OCR](#deepseek-ocr-visual-context-compression-through-deepencoder) | [PaddleOCR-VL](#paddleocr-vl-ultra-compact-multilingual-document-parsing) | [Qwen3-VL](#qwen3-vl-deepstack-vision-language-models) | [Step3](#step3-model-system-co-design-for-cost-effective-multimodal-intelligence) | [GLM-4.1V-Thinking](#glm-41v-thinking-general-purpose-multimodal-reasoning-through-curriculum-sampled-rl) | [ERNIE 4.5-VL](#ernie-45-vl-heterogeneous-modality-mixture-of-experts) | [MiMo-VL](#mimo-vl-multimodal-pretraining-with-mixed-on-policy-reinforcement-learning) | [BAGEL](#bagel-a-mixture-of-transformer-experts-for-unified-understanding-and-generation) | [Seed1.5-VL](#seed15-vl-sparse-moe-multimodal-understanding-and-agentic-reasoning) | [InternVL3 / 3.5](#internvl3-and-internvl35-native-multimodal-pretraining-and-adaptive-resolution) | [Kimi-VL](#kimi-vl-native-resolution-vision-with-a-sparse-moe-decoder) | [Llama 4](#llama-4-scout-and-maverick-native-multimodal-mixture-of-experts-models) | [Qwen2.5-Omni](#qwen25-omni-streaming-multimodal-perception-and-speech-generation) | [Gemma 3](#gemma-3-long-context-multimodality-with-efficient-interleaved-attention) | [Aya Vision](#aya-vision-multilingual-multimodality-through-cross-modal-model-merging) | [Phi-4-multimodal](#phi-4-multimodal-text-vision-and-speech-through-mixture-of-loras) | [SigLIP 2](#siglip-2-multilingual-vision-language-encoders-with-native-aspect-ratio-support)

**2024 additions:** [ShowUI](#showui-vision-language-action-modeling-for-gui-agents) | [Janus / Janus-Pro](#janus-and-janus-pro-decoupled-visual-understanding-and-generation) | [Emu3](#emu3-next-token-prediction-across-text-image-and-video) | [Molmo / PixMo](#molmo-and-pixmo-open-weights-open-data-and-grounded-pointing) | [VILA-U](#vila-u-fully-autoregressive-visual-understanding-and-generation) | [Show-o](#show-o-autoregressive-language-and-discrete-diffusion-vision-in-one-transformer) | [Transfusion](#transfusion-next-token-text-prediction-and-continuous-image-diffusion) | [mPLUG-Owl3](#mplug-owl3-hyper-attention-for-long-image-sequences) | [Cambrian-1](#cambrian-1-vision-centric-multimodal-llms) | [Ovis](#ovis-structural-visual-text-embedding-alignment) | [Phi-3-Vision](#phi-3-vision-and-phi-35-vision-compact-long-context-multimodal-reasoning) | [Chameleon](#chameleon-mixed-modal-early-fusion-foundation-models) | [MM1](#mm1-methods-analysis-and-insights-from-multimodal-pre-training) | [AnyGPT](#anygpt-unified-any-to-any-multimodal-modeling-with-discrete-tokens)

[LLaVA](#llava-large-language-and-vision-assistant---visual-instruction-tuning) | [LLaVA 1.5](#llava-15-improved-baselines-with-visual-instruction-tuning) | [LLaVA 1.6](#llava-16-llava-next-improved-reasoning-ocr-and-world-knowledge) | [PaliGemma](#paligemma-a-versatile-and-transferable-3b-vision-language-model) | [PaliGemma 2](#paligemma-2-a-family-of-versatile-vlms-for-transfer) | [AIMv2](#aimv2-multimodal-autoregressive-pre-training-of-large-vision-encoders) | [Apollo](#apollo-an-exploration-of-video-understanding-in-large-multimodal-models) | [ARIA](#aria-an-open-multimodal-native-mixture-of-experts-model) | [EVE](#eve-unveiling-encoder-free-vision-language-models) | [EVEv2](#evev2-improved-baselines-for-encoder-free-vision-language-models) | [Janus / Janus-Pro](#janus-and-janus-pro-decoupled-visual-understanding-and-generation) | [LLaVA-CoT](#llava-cot-let-vision-language-models-reason-step-by-step) | [LLM2CLIP](#llm2clip-powerful-language-model-unlocks-richer-visual-representation) | [Maya](#maya-an-instruction-finetuned-multilingual-multimodal-model) | [MiniMax-01](#minimax-01-scaling-foundation-models-with-lightning-attention) | [NVLM](#nvlm-open-frontier-class-multimodal-llms) | [OmniVLM](#omnivlm-a-token-compressed-sub-billion-parameter-vision-language-model-for-efficient-on-device-inference) | [Pixtral 12B](#pixtral-12b-a-cutting-edge-open-multimodal-language-model) | [Sa2VA](#sa2va-marrying-sam2-with-llava-for-dense-grounded-understanding-of-images-and-videos) | [Tarsier2](#tarsier2-advancing-large-vision-language-models-from-detailed-video-description-to-comprehensive-video-understanding) | [UI-TARS](#ui-tars-pioneering-automated-gui-interaction-with-native-agents) | [VideoChat-Flash](#videochat-flash-hierarchical-compression-for-long-context-video-modeling) | [VideoLLaMA 3](#videollama-3-frontier-multimodal-foundation-models-for-image-and-video-understanding) | [Llama 3.2-Vision](#llama-32-vision-enhanced-multimodal-capabilities-built-on-llama-3) | [SmolVLM](#smolvlm-a-small-efficient-and-open-source-vision-language-model) | [IDEFICS](#idefics) | [IDEFICS2](#idefics2) | [IDEFICS3-8B](#idefics3-8b-building-and-better-understanding-vision-language-models) | [InternLM-XComposer2](#internlm-xcomposer2-mastering-free-form-text-image-composition-and-comprehension-in-vision-language-large-model) | [InternLM-XComposer2-4KHD](#internlm-xcomposer2-4khd-a-pioneering-large-vision-language-model-handling-resolutions-from-336-pixels-to-4k-hd) | [InternLM-XComposer-2.5](#internlm-xcomposer-25-a-versatile-large-vision-language-model-supporting-long-contextual-input-and-output) | [InternVL 2.5](#internvl-25-expanding-performance-boundaries-of-open-source-multimodal-models-with-model-data-and-test-time-scaling) | [DeepSeek-VL](#deepseek-vl-towards-real-world-vision-language-understanding) | [DeepSeek-VL2](#deepseek-vl2-mixture-of-experts-vision-language-models-for-advanced-multimodal-understanding) | [MANTIS](#mantis-mastering-multi-image-understanding-through-interleaved-instruction-tuning) | [Qwen-VL](#qwen-vl-a-versatile-vision-language-model-for-understanding-localization-text-reading-and-beyond) | [Qwen2-VL](#qwen2-vl-a-powerful-open-source-vision-language-model-for-image-and-video-understanding) | [Qwen2.5-VL](#qwen25-vl-enhanced-vision-language-capabilities-in-the-qwen-series) | [moondream1](#moondream1-and-moondream2) | [moondream2](#moondream1-and-moondream2) | [Moondream-next](#moondream-next-compact-vision-language-model-with-enhanced-capabilities) | [SPHINX-X](#sphinx-x-scaling-data-and-parameters-for-a-family-of-multi-modal-large-language-models) | [BLIP](#blip-bootstrapping-language-image-pre-training) | [BLIP-2](#blip-2-bootstrapping-language-image-pre-training-with-frozen-image-encoders-and-large-language-models) | [xGen-MM (BLIP-3)](#xgen-mm-blip-3-an-open-source-framework-for-building-powerful-and-responsible-large-multimodal-models) | [InstructBLIP](#instructblip-towards-general-purpose-vision-language-models-with-instruction-tuning) | [KOSMOS-1](#kosmos-1-language-is-not-all-you-need-aligning-perception-with-language-models) | [KOSMOS-2](#kosmos-2-grounding-multimodal-large-language-models-to-the-world) | [ConvLLaVA](#convllava-hierarchical-backbones-as-visual-encoder-for-large-multimodal-models) | [Parrot](#parrot-multilingual-visual-instruction-tuning) | [OMG-LLaVA](#omg-llava-bridging-image-level-object-level-pixel-level-reasoning-and-understanding) | [EVLM](#evlm-an-efficient-vision-language-model-for-visual-understanding) | [SlowFast-LLaVA](#slowfast-llava-a-strong-training-free-baseline-for-video-large-language-models) | [Nous-Hermes-2-Vision - Mistral 7B](#nous-hermes-2-vision---mistral-7b) | [TinyGPT-V](#tinygpt-v-efficient-multimodal-large-language-model-via-small-backbones) | [CoVLM](#covlm-composing-visual-entities-and-relationships-in-large-language-models-via-communicative-decoding) | [GLaMM](#glamm-pixel-grounding-large-multimodal-model) | [COSMO](#cosmo-contrastive-streamlined-multimodal-model-with-interleaved-pre-training) | [FireLLaVA](#firellava) | [u-LLaVA](#u-llava-unifying-multi-modal-tasks-via-large-language-model) | [MoE-LLaVA](#moe-llava-mixture-of-experts-for-large-vision-language-models) | [BLIVA](#bliva-a-simple-multimodal-llm-for-better-handling-of-text-rich-visual-questions) | [MobileVLM](#mobilevlm-a-fast-strong-and-open-vision-language-assistant-for-mobile-devices) | [FROZEN](#frozen-multimodal-few-shot-learning-with-frozen-language-models) | [Flamingo](#flamingo-a-visual-language-model-for-few-shot-learning) | [OpenFlamingo](#openflamingo-an-open-source-framework-for-training-large-autoregressive-vision-language-models) | [PaLI](#pali-a-jointly-scaled-multilingual-language-image-model) | [PaLI-3](#pali-3-vision-language-models-smaller-faster-stronger) | [PaLM-E](#palm-e-an-embodied-multimodal-language-model) | [MiniGPT-4](#minigpt-4-enhancing-vision-language-understanding-with-advanced-large-language-models) | [MiniGPT-v2](#minigpt-v2-large-language-model-as-a-unified-interface-for-vision-language-multi-task-learning) | [LLaVA-Plus](#llava-plus-learning-to-use-tools-for-creating-multimodal-agents) | [BakLLaVA](#bakllava) | [CogVLM](#cogvlm-visual-expert-for-pretrained-language-models) | [CogVLM2](#cogvlm2-enhanced-vision-language-models-for-image-and-video-understanding) | [Ferret](#ferret-refer-and-ground-anything-anywhere-at-any-granularity) | [Fuyu-8B](#fuyu-8b-a-multimodal-architecture-for-ai-agents) | [OtterHD](#otterhd-a-high-resolution-multi-modality-model) | [SPHINX](#sphinx-the-joint-mixing-of-weights-tasks-and-visual-embeddings-for-multi-modal-large-language-models) | [Eagle 2](#eagle-2-building-post-training-data-strategies-from-scratch-for-frontier-vision-language-models) | [EAGLE](#eagle-exploring-the-design-space-for-multimodal-llms-with-mixture-of-encoders) | [VITA](#vita-towards-open-source-interactive-omni-multimodal-llm) | [LLaVA-OneVision](#llava-onevision-easy-visual-task-transfer) | [MiniCPM-o-2.6](#minicpm-o-26-a-gpt-4o-level-mllm-for-vision-speech-and-multimodal-live-streaming) | [MiniCPM-V](#minicpm-v-a-gpt-4v-level-mllm-on-your-phone) | [INF-LLaVA](#inf-llava-high-resolution-image-perception-for-multimodal-large-language-models) | [Florence-2](#florence-2-a-deep-dive-into-its-unified-architecture-and-multi-task-capabilities) | [MULTIINSTRUCT](#multiinstruct-improving-multi-modal-zero-shot-learning-via-instruction-tuning) | [MouSi](#mousi-poly-visual-expert-vision-language-models) | [LaVIN](#lavin-cheap-and-quick-efficient-vision-language-instruction-tuning-for-large-language-models) | [CLIP](#clip-contrastive-language-image-pre-training) | [MetaCLIP](#metaclip-demystifying-clip-data) | [Alpha-CLIP](#alpha-clip-a-clip-model-focusing-on-wherever-you-want) | [GLIP](#glip-grounded-language-image-pre-training) | [ImageBind](#imagebind-one-embedding-space-to-bind-them-all) | [SigLIP](#siglip-sigmoid-loss-for-language-image-pre-training) | [ViT](#vit-an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale)

## Release Timeline

Dates below use the first official model release when one is documented; otherwise they use the paper's arXiv v1 submission or the first technical report. Point releases are folded into their architecture family.

<!--lint disable table-pipe-alignment table-cell-padding-->

| Date | Architecture | Distinctive contribution |
|---|---|---|
| 2026-07-28 | MODUS | Decoder-only any-to-any modeling without modality-specific heads or losses |
| 2026-07-28 | Argus-Unified | Hybrid continuous and discrete visual tokens for economical understanding and generation |
| 2026-07-27 | Kimi K3 | Kimi Delta Attention, Attention Residuals, and extremely sparse LatentMoE routing |
| 2026-07-27 | Mage-VL | Codec-native selective video tokenization with a proactive event gate |
| 2026-07-15 | Inkling | Relative-position million-context multimodal MoE trained from scratch |
| 2026-07-15 | Hy-Embodied-VLM | Action-centric sparse-MoE reasoning for physical-world agents |
| 2026-07-11 | MonkeyOCRv2 | Joint image-to-text and pixel-reconstruction pretraining for document vision |
| 2026-06-11 | MiniMax M3 | Native multimodality with block-sparse grouped-query attention at million-token context |
| 2026-06-10 | InternVideo3 | Token-preserving latent KV compression and closed-loop video reasoning |
| 2026-06-09 | Keye-VL 2.0 | DeepSeek Sparse Attention adapted to GQA-based long-video multimodality |
| 2026-06-02 | Zamba2-VL | Hybrid Mamba-2 and shared-attention blocks for efficient VLM inference |
| 2026-05-31 | Cosmos 3 | Coupled autoregressive reasoner and diffusion generator for physical AI |
| 2026-05-18 | Lance | Shared-sequence understanding, generation, and editing with modality experts |
| 2026-05-08 | ZAYA1-VL | Vision-conditional LoRA and compressed convolutional attention in an open-data MoE |
| 2026-05-03 | Falcon Perception | Early fusion with hybrid attention and continuous mask heads |
| 2026-04-29 | GLM-5V-Turbo | Perception integrated into reasoning, planning, tools, and execution |
| 2026-04-21 | PLaMo 2.1-VL | Compact Japanese VQA and grounding for edge deployment |
| 2026-04-09 | EXAONE 4.5 | Native multimodal pretraining with document-focused data and 256K context |
| 2026-04-02 | BidirLM / BidirLM-Omni | Converting causal decoders into bidirectional multimodal encoders |
| 2026-03-31 | Gemma 4 | Dense and MoE native multimodality, including an encoder-free 12B design |
| 2026-03-06 | Penguin-VL | Text-LLM-initialized vision encoder and priority-aware token compression |
| 2026-03-04 | Phi-4-Reasoning-Vision | Mid-fusion compact VLM with explicit reasoning and direct-answer modes |
| 2026-03-01 | V-SONAR / V-LCM | Vision-language alignment and prediction in multilingual concept space |
| 2026-02-16 | Qwen3.5 | Native early fusion with hybrid linear/full attention and sparse MoE variants |
| 2026-01-27 | Youtu-VL | Unified autoregressive visual tokens that emit dense vision outputs without task heads |
| 2026-01-27 | Kimi K2.5 / K2.6 | Trillion-parameter native multimodal MoE for agents and computer use |
| 2026-01-14 | Step3-VL-10B | Language-aligned perception encoder with 16-fold visual-token compression |
| 2025-11-13 | ERNIE 5.0 | One autoregressive sparse MoE for text, images, video, audio, and generation |
| 2025-10-20 | DeepSeek-OCR | DeepEncoder compresses high-resolution documents into very short visual contexts |
| 2025-10-16 | PaddleOCR-VL | NaViT-style dynamic resolution with a compact ERNIE decoder for document parsing |
| 2025-09-22 | Qwen3-VL | DeepStack multi-level ViT fusion and explicit video timestamp alignment |
| 2025-07-25 | Step3 | Model-system co-design for communication-efficient sparse-MoE multimodality |
| 2025-07-01 | GLM-4.1V-Thinking | Curriculum-sampled reinforcement learning for multimodal reasoning |
| 2025-06-30 | ERNIE 4.5-VL | Heterogeneous shared and modality-specific experts with isolated routing |
| 2025-06-04 | MiMo-VL | Four-stage multimodal pretraining followed by mixed on-policy RL |
| 2025-05-20 | BAGEL | Mixture-of-Transformer-Experts for understanding and generation |
| 2025-05-11 | Seed1.5-VL | Compact vision encoder with a 20B-active MoE for reasoning and agents |
| 2025-04-11 | InternVL3 | Native multimodal pretraining, later extended with adaptive resolution and cascade RL |
| 2025-04-10 | Kimi-VL | MoonViT native-resolution packing with a sparse MoE decoder |
| 2025-04-05 | Llama 4 | Early-fusion native multimodality in sparse-MoE Scout and Maverick models |
| 2025-03-26 | Qwen2.5-Omni | Streaming Thinker-Talker architecture for multimodal input and speech output |
| 2025-03-12 | Gemma 3 | Efficient local/global attention with long-context image understanding |
| 2025-03-04 | Aya Vision | Cross-modal model merging for multilingual multimodality without language forgetting |
| 2025-03-03 | Phi-4-multimodal | Mixture-of-LoRAs for text, vision, and speech |
| 2025-02-20 | SigLIP 2 | Multilingual, localization-aware, native-aspect-ratio vision-language encoding |
| 2024-11-26 | ShowUI | UI-guided visual-token selection and interleaved action histories |
| 2024-10-17 | Janus | Decoupled visual encoders for understanding and generation with one transformer |
| 2024-09-27 | Emu3 | One next-token objective over discrete text, image, and video tokens |
| 2024-09-25 | Molmo / PixMo | Open data pipeline with human captions and grounded pointing supervision |
| 2024-09-06 | VILA-U | Shared discrete visual tokens for autoregressive understanding and generation |
| 2024-08-22 | Show-o | Autoregressive language and discrete-diffusion image generation in one transformer |
| 2024-08-20 | Transfusion | Autoregressive text and continuous image diffusion in one transformer |
| 2024-08-09 | mPLUG-Owl3 | Hyper-attention for long image sequences and video |
| 2024-06-24 | Cambrian-1 | Spatial Vision Aggregator and systematic multi-encoder study |
| 2024-06-14 | Ovis | Learnable visual vocabulary for structural visual-text embedding alignment |
| 2024-05-21 | Phi-3-Vision | Compact dynamic-resolution VLM with 128K context |
| 2024-05-16 | Chameleon | Mixed-modal early fusion over a shared token sequence |
| 2024-03-14 | MM1 | Controlled study of encoders, connectors, token counts, and data mixtures |
| 2024-02-19 | AnyGPT | Any-to-any autoregression over discrete text, image, speech, and music tokens |

<!--lint enable table-pipe-alignment table-cell-padding-->

## Architectures

### **MODUS: Decoder-Only Any-to-Any Multimodal Modeling**

MODUS treats every modality symmetrically as both input and output, enabling chained generation and cross-modal self-verification without modality-specific heads, losses, or task pipelines.

[![arXiv](https://img.shields.io/badge/arXiv-2607.25948-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.25948) [![Project](https://img.shields.io/badge/Project-MODUS-blue?style=flat-square)](https://modus-multimodal.epfl.ch/)

Mingqiao Ye et al., EPFL<br>
**Released:** 2026-07-28

<details>
<summary>ℹ️ <i>More Information</i></summary>

**MODUS** is a decoder-only any-to-any model that represents diverse modalities inside one autoregressive architecture. Unlike encoder-decoder or diffusion systems assembled around modality-specific output paths, the same model predicts any supported modality from any combination of the others. This makes intermediate-modality chains and self-scoring through a second generated modality native behaviors rather than external workflows.

The design deliberately reuses strong pretrained decoder-only priors instead of training a bespoke multimodal stack from scratch. A single checkpoint is evaluated across heterogeneous tasks and modalities, making MODUS most notable as a general architectural formulation rather than a narrowly optimized VLM endpoint.

</details>

### **Argus-Unified: Economical Understanding and Generation**

Argus-Unified combines continuous tokens for understanding with learned discrete tokens for generation, reusing a frozen unified visual encoder and pretrained VLM to lower the cost of unified modeling.

[![arXiv](https://img.shields.io/badge/arXiv-2607.25527-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.25527)

Weiming Zhuang et al.<br>
**Released:** 2026-07-28

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Argus-Unified** resolves the conflicting visual representations required by comprehension and synthesis through hybrid visual tokens. Continuous encoder features preserve semantic information for understanding, while a learned quantizer produces discrete tokens for image generation from the same frozen visual encoder.

Training first learns the quantizer and image decoder, then initializes the language component from a pretrained VLM and trains unified multimodal prediction. The reported recipe uses 15.6 million examples and roughly $2,000 of compute, positioning the model as a reproducible baseline for understanding-and-generation research rather than a scale demonstration.

</details>

### **Kimi K3: Kimi Delta Attention at Trillion-Parameter Scale**

Kimi K3 is a native multimodal sparse MoE that combines Kimi Delta Attention, gated latent attention, Attention Residuals, and 896 routed experts for million-token reasoning and agency.

[![arXiv](https://img.shields.io/badge/arXiv-2607.24653-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.24653) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/MoonshotAI/Kimi-K3) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/moonshotai/Kimi-K3)

Kimi Team, Moonshot AI<br>
**Released:** 2026-07-27

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Kimi K3** has 2.8T total and 104B active parameters across 93 layers. Sixty-nine layers use gated delta-rule linear attention and 24 use gated Multi-head Latent Attention; learned Attention Residuals create structured information paths across depth. Stable LatentMoE activates 16 of 896 routed experts plus two shared experts, making the model unusually sparse for its scale.

A 401M-parameter MoonViT-V2 encoder supplies native image and video input. The model supports a 1,048,576-token context and receives multimodal, long-context, agentic, reasoning, and tool-use training. Quantization-aware pretraining targets MXFP4 weights and MXFP8 activations rather than treating low-precision deployment as an afterthought.

</details>

### **Mage-VL: Codec-Native Streaming Multimodality**

Mage-VL uses video-codec signals to encode only dynamic, information-rich regions and pairs a lightweight event gate with a causal decoder for proactive streaming perception.

[![arXiv](https://img.shields.io/badge/arXiv-2607.24904-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.24904) [![Project](https://img.shields.io/badge/Project-Mage--VL-blue?style=flat-square)](https://microsoft.github.io/Mage/)

Senqiao Yang et al., Microsoft Research<br>
**Released:** 2026-07-27

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Mage-ViT** replaces uniform frame sampling with selective 16×16-patch encoding guided by motion vectors and residual energy across anchor and predicted frames. This codec-native tokenization reduces visual-token use by more than 75% while retaining spatiotemporal context. The encoder is trained from scratch on roughly 560M unlabeled images and 100M unlabeled video frames.

The complete VLM uses a dual-system design: a small event gate decides when a stream warrants deeper processing, and a causal decoder performs contextual reasoning. This turns streaming perception into an event-driven process and yields up to a reported 3.5× wall-clock speedup while retaining static-image and spatial reasoning ability.

</details>

### **Inkling: Relative-Position Multimodal Mixture of Experts**

Inkling is an open-weight multimodal MoE trained from scratch on text, images, audio, and video, using relative positions and alternating local/global attention for million-token contexts.

[![Website](https://img.shields.io/badge/Official-Inkling-blue?style=flat-square)](https://thinkingmachines.ai/news/introducing-inkling/) [![Model Card](https://img.shields.io/badge/Model_Card-Inkling-blue?style=flat-square)](https://thinkingmachines.ai/model-card/inkling/) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/thinkingmachines/Inkling)

Thinking Machines Lab<br>
**Released:** 2026-07-15

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Inkling** contains 975B total and 41B active parameters. Its MoE layers activate six of 256 routed experts plus two shared experts using sigmoid routing and auxiliary-loss-free balancing. Attention alternates five local sliding-window layers with one global layer and uses learned relative-position representations rather than RoPE; short convolutions further refine key, value, attention, and MLP pathways.

Pretraining spans 45T text, image, audio, and video tokens. Post-training begins with a small synthetic supervised bootstrap and scales asynchronous reinforcement learning across multimodal understanding, tools, coding, mathematics, conversation, and safety. The released weights support controllable reasoning effort and contexts up to one million tokens.

</details>

### **Hy-Embodied-VLM: Sparse-MoE Reasoning for Physical Agents**

Hy-Embodied-VLM-1.0 joins a native-aspect-ratio vision encoder to a compact sparse MoE and trains an action-centric reasoning hierarchy for perception, planning, reflection, and recovery.

[![arXiv](https://img.shields.io/badge/arXiv-2607.12894-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.12894) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Tencent-Hunyuan/HY-Embodied) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0)

Tencent Robotics X, Hy Vision Team and Futian Laboratory<br>
**Released:** 2026-07-15

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Hy-Embodied-VLM-1.0** connects Hy-ViT2 to a roughly 30B-total, 3B-active language model. Each sparse-MoE layer selects eight of 128 routed experts plus one shared expert. It accepts as many as 128 images in a 32K context and exposes direct-response and explicit-thinking modes through the same checkpoint.

Its curriculum moves from action-relevant state understanding to transition reasoning and then sequential, adaptive reasoning. A self-evolving post-training loop alternates reinforcement learning with rejection-sampling fine-tuning, separately optimizing geometric precision and higher-level planning before policy fusion.

</details>

### **MonkeyOCRv2: Document-Native Visual-Text Pretraining**

MonkeyOCRv2 learns document vision jointly through image-to-text generation and pixel reconstruction, preserving both semantic content and character-level strokes across 17 languages.

[![arXiv](https://img.shields.io/badge/arXiv-2607.11562-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.11562) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/Yuliang-Liu/MonkeyOCRv2)

Yuliang Liu et al.<br>
**Released:** 2026-07-11

<details>
<summary>ℹ️ <i>More Information</i></summary>

**MonkeyOCRv2** is a visual-text foundation encoder specialized for the properties that natural-image pretraining often misses: dense text, small glyphs, fine strokes, and layout. Its two complementary objectives align images with their transcribed content while forcing the encoder to retain pixel-level document structure.

The MonkeyDoc v2 corpus contains 113M document images in 17 languages. The frozen encoder improves recognition, detection, tamper analysis, segmentation, parsing, and understanding; paired with a lightweight language model, it forms a 0.7B document parser. Public weights were announced July 11, two days before the arXiv report, so the timeline follows the model release rather than the paper date.

</details>

### **MiniMax M3: Native Multimodality with Sparse Long-Context Attention**

MiniMax M3 combines native multimodal pretraining with learned block-sparse grouped-query attention, making million-token contexts practical in a 109B-parameter model.

[![arXiv](https://img.shields.io/badge/arXiv-2606.13392-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.13392) [![Website](https://img.shields.io/badge/Official-MiniMax_M3-blue?style=flat-square)](https://www.minimax.io/blog/minimax-m3) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/MiniMaxAI/MiniMax-M3)

MiniMax<br>
**Released:** 2026-06-11

<details>
<summary>ℹ️ <i>More Information</i></summary>

**MiniMax Sparse Attention** adds a lightweight index branch that scores KV blocks independently for each grouped-query-attention group. The main branch performs exact attention only over selected blocks, while a KV-outer kernel groups queries retrieving the same block to improve memory locality and reuse.

The released 109B-parameter model is trained natively across modalities and evaluated at contexts up to one million tokens. The sparse-attention report focuses on architecture, kernel design, and scaling behavior; it does not publish a complete source-level training-data inventory.

</details>

### **InternVideo3: Multimodal Contextual Reasoning for Video Agents**

InternVideo3 reframes long-video understanding as an evolving loop of observation, reasoning, tools, and memory, while compressing KV state without dropping the underlying token stream.

[![arXiv](https://img.shields.io/badge/arXiv-2606.12195-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.12195)

Ziang Yan et al.<br>
**Released:** 2026-06-10

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Multimodal Contextual Reasoning** maintains one evolving context containing observations, instructions, intermediate reasoning, tool actions, and memory. This closed loop lets the model accumulate and verify evidence across long videos rather than treating video QA as a single static prompt.

**Multimodal Multi-head Latent Attention** reparameterizes and compresses KV-cache states while preserving the complete visual-token sequence. Training combines continued pretraining, short-to-long supervised tuning, rule-based reinforcement learning, and on-policy distillation; a retrieval-equipped video-agent implementation demonstrates the intended tool-using setting.

</details>

### **Keye-VL 2.0: Sparse Attention for Long-Video Agents**

Keye-VL 2.0 adapts DeepSeek Sparse Attention to a GQA-based multimodal MoE, targeting lossless 256K contexts, hour-scale video, and self-correcting tool-using agents.

[![arXiv](https://img.shields.io/badge/arXiv-2606.10651-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.10651)

Kwai Keye Team<br>
**Released:** 2026-06-09

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Keye-VL-2.0-30B-A3B** is the first reported adaptation of DeepSeek Sparse Attention to grouped-query multimodal attention. Only 3B of 30B parameters activate per token, while the sparse retrieval mechanism keeps critical frames and long-range dependencies accessible across a 256K context. Heterogeneous ViT-language-model parallelism and custom sparse-attention kernels address the systems cost of hour-long video.

Cross-Modal Multi-Teacher On-Policy Distillation feeds dense token-level guidance from specialist teachers back into on-policy trajectories. Context-RL and Video-RL then target long-context reasoning, temporal localization, code, search, tools, and multimodal self-correction without collapsing the multi-task mixture.

</details>

### **Zamba2-VL: Hybrid State-Space Vision-Language Modeling**

Zamba2-VL combines efficient Mamba-2 state-space layers with a small number of shared Transformer blocks, reducing long-context prefill and recurrent-state costs across compact VLM scales.

[![arXiv](https://img.shields.io/badge/arXiv-2606.00390-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.00390) [![Project](https://img.shields.io/badge/Official-Zamba2--VL-blue?style=flat-square)](https://www.zyphra.com/our-work/zamba2-vl)

Zyphra<br>
**Released:** 2026-06-02

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Zamba2-VL** extends the Zamba hybrid backbone to vision-language inputs in 1.2B, 2.7B, and 7B variants. Most sequence processing occurs in Mamba-2 state-space layers, with a small set of shared attention blocks supplying global token interaction. This preserves transformer-like multimodal reasoning while moving more computation toward near-linear prefill and bounded recurrent state.

The architecture is particularly relevant for edge and long-context deployment: visual sequences can be large even when the language model is small, so reducing quadratic attention changes time-to-first-token and memory behavior more materially than ordinary parameter pruning.

</details>

### **Cosmos 3: Omnimodal World Modeling with Mixture of Transformers**

Cosmos 3 couples an autoregressive reasoner with a diffusion generator through a shared representation for language, vision, audio, actions, simulation, and robot control.

[![arXiv](https://img.shields.io/badge/arXiv-2606.02800-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.02800) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/NVIDIA/cosmos) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/nvidia/cosmos3)

NVIDIA<br>
**Released:** 2026-05-31

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Cosmos 3** uses a Mixture-of-Transformers with two communicating towers. An autoregressive transformer reasons over language, vision, and action sequences; a diffusion transformer generates continuous images, video, and world trajectories. Their shared representation lets the family serve as a VLM, forward or inverse dynamics model, simulator, generator, or policy without separate task pipelines.

Nano configurations combine 8B-class components and Super configurations combine 32B-class components. Training spans text, image, video, audio, simulated physical interaction, action trajectories, robot demonstrations, driving, and synthetic worlds. The July 20 Cosmos 3 Edge release preserves the same two-tower design in a smaller real-time deployment branch and is folded into this family.

</details>

### **Lance: Unified Image and Video Understanding, Generation, and Editing**

Lance is a compact model trained from scratch to understand, generate, and edit images and video inside one interleaved sequence while preserving specialized semantic and synthesis capacity.

[![arXiv](https://img.shields.io/badge/arXiv-2605.18678-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2605.18678) [![Project](https://img.shields.io/badge/Project-Lance-blue?style=flat-square)](https://lance-project.github.io/)

Lance Team<br>
**Released:** 2026-05-18

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Lance** interleaves text, image, and video in a shared 3B-parameter sequence model while routing semantic understanding and visual synthesis through dedicated experts. Semantic ViT tokens coexist with clean or noised VAE latents, and generalized 3D causal attention plus modality-aware positional encoding preserve spatial and temporal relationships.

One checkpoint supports visual understanding, text-to-image and text-to-video generation, image-to-video generation, and editing. Its training-from-scratch recipe uses no more than 128 A100 GPUs, making the contribution as much about an attainable unified architecture as headline generation quality.

</details>

### **ZAYA1-VL: Vision-Specialized Compressed Convolutional Attention**

ZAYA1-VL adds visual capacity to a sparse MoE through vision-only LoRA and compressed-convolutional-attention parameters, trained on an entirely open multimodal data mixture.

[![arXiv](https://img.shields.io/badge/arXiv-2605.08560-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2605.08560) [![Project](https://img.shields.io/badge/Official-ZAYA1--VL-blue?style=flat-square)](https://www.zyphra.com/our-work/zaya1-vl-8b) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/Zyphra/ZAYA1-VL-8B)

Zyphra<br>
**Released:** 2026-05-08

<details>
<summary>ℹ️ <i>More Information</i></summary>

**ZAYA1-VL-8B** joins a Qwen2.5-VL visual encoder to ZAYA1's MoE decoder. Vision-only LoRA and Compressed Convolutional Attention parameters activate for visual tokens without duplicating the language backbone, while bidirectional attention within image-token spans improves spatial integration.

The model is trained on roughly 140B vision-language tokens assembled from open data and released under Apache 2.0. Its importance is the clean separation of visual specialization from shared language capacity in a compact sparse model.

</details>

### **Falcon Perception: Early-Fusion Grounding, Segmentation, and OCR**

Falcon Perception uses early image-text fusion, hybrid attention, instance tokens, and continuous mask heads to unify open-vocabulary grounding, segmentation, and OCR in compact models.

[![arXiv](https://img.shields.io/badge/arXiv-2603.27365-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.27365) [![Project](https://img.shields.io/badge/Project-Falcon_Perception-blue?style=flat-square)](https://vision.falcon.aidrc.tii.ae/) [![Release](https://img.shields.io/badge/Official-Release-blue?style=flat-square)](https://www.tii.ae/news/tii-launches-falcon-perception-new-multimodal-ai-model-helps-machines-see-and-understand-world)

Technology Innovation Institute<br>
**Released:** 2026-05-03

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Falcon Perception** is a dense early-fusion Transformer in which image patches and text interact from the first layer. Its attention mask is bidirectional over image tokens and causal over prediction tokens. Variable-length instance tokens identify multiple objects, while lightweight continuous heads produce masks without turning segmentation into an unwieldy text sequence.

The family includes a 600M perception model and a 300M OCR model. The paper was submitted March 28, but the timeline uses the documented May 3 public model launch, following this repository's release-date policy.

</details>

### **GLM-5V-Turbo: Native Multimodal Agency**

GLM-5V-Turbo integrates visual perception into reasoning, planning, tool use, execution, and verification instead of treating images and video as an auxiliary language-model interface.

[![arXiv](https://img.shields.io/badge/arXiv-2604.26752-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.26752)

GLM-V Team<br>
**Released:** 2026-04-29

<details>
<summary>ℹ️ <i>More Information</i></summary>

**GLM-5V-Turbo** is organized around end-to-end multimodal agency across images, video, webpages, documents, GUIs, code, and external tools. Its contribution is the integration of perception with the agent loop: observations remain available during planning and execution, and outcomes can be visually verified rather than handed off to a detached captioning stage.

The report describes coordinated changes to model design, multimodal training, hierarchical optimization, reinforcement learning, toolchains, and agent-framework integration. Because this is a new native agent foundation model rather than a latency-only checkpoint, it is kept separate from the GLM-4.1V family.

</details>

### **PLaMo 2.1-VL: Lightweight Japanese Vision-Language Modeling**

PLaMo 2.1-VL provides compact 2B and 8B Japanese-English VLMs that combine visual question answering and grounding for local, edge, and autonomous-device deployment.

[![arXiv](https://img.shields.io/badge/arXiv-2604.19324-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.19324)

Tommi Kerola et al., Preferred Networks<br>
**Released:** 2026-04-21

<details>
<summary>ℹ️ <i>More Information</i></summary>

**PLaMo 2.1-VL** is designed around a constrained deployment envelope rather than frontier scale. Both sizes support VQA and bounding-box grounding, with evaluations emphasizing Japanese language operation and applications such as factory tool recognition and infrastructure anomaly detection.

A large synthetic-data pipeline supplies Japanese multimodal instructions, grounding examples, and domain-oriented supervision. The entry broadens the catalog's language and edge coverage; its contribution is a deployable bilingual VQA-and-grounding family rather than a new attention primitive.

</details>

### **EXAONE 4.5: Native Multimodal Pretraining for Documents**

EXAONE 4.5 is LG AI Research's first open-weight VLM, integrating a dedicated visual encoder into EXAONE and emphasizing document-centric native multimodal pretraining and Korean context.

[![arXiv](https://img.shields.io/badge/arXiv-2604.08644-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.08644)

Eunbi Choi et al., LG AI Research<br>
**Released:** 2026-04-09

<details>
<summary>ℹ️ <i>More Information</i></summary>

**EXAONE 4.5** adds a dedicated vision encoder to the EXAONE 4.0 framework and trains the resulting stack natively over text and visual inputs. Its data mixture is deliberately document-heavy, targeting enterprise documents, OCR-rich layouts, and Korean contextual reasoning while retaining general language capability.

Context extends to 256K tokens for long documents and enterprise workflows. The combination of open weights, native multimodal training, long context, and focused document/Korean coverage makes it a distinct public VLM release rather than a post-training adapter.

</details>

### **BidirLM and BidirLM-Omni: Causal Decoders as Multimodal Encoders**

BidirLM converts pretrained causal language decoders into bidirectional representation encoders; BidirLM-Omni extends the method to one contrastively aligned text, image, and audio embedding model.

[![arXiv](https://img.shields.io/badge/arXiv-2604.02045-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2604.02045) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/BidirLM) [![Release](https://img.shields.io/badge/Official-Release-blue?style=flat-square)](https://huggingface.co/blog/Nicolas-BZRD/bidirlm-release)

BidirLM Team<br>
**Released:** 2026-04-02

<details>
<summary>ℹ️ <i>More Information</i></summary>

**BidirLM** removes the causal mask and adapts a decoder with masked next-token prediction and contrastive learning. Weight merging and mixed-domain training preserve generative knowledge while recovering the full-context interaction needed by retrieval and embedding tasks.

**BidirLM-Omni** combines Qwen3 text, Qwen3-VL, and Qwen3-ASR specialist backbones with frozen visual and audio projectors, aligning all three modalities in one contrastive space. The entry belongs beside CLIP, SigLIP, and ImageBind as an encoder architecture, despite originating from decoder checkpoints.

</details>

### **Gemma 4: Open-Weight Native Multimodal Models**

Gemma 4 is Google DeepMind's open-weight multimodal family spanning mobile-efficient, dense, sparse-MoE, and encoder-free designs, with configurable reasoning and deployment targets ranging from phones to servers.

[![arXiv](https://img.shields.io/badge/arXiv-2607.02770-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2607.02770) [![Releases](https://img.shields.io/badge/Official-Releases-blue?style=flat-square)](https://ai.google.dev/gemma/docs/releases) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/google/gemma-4)

Gemma Team, Google DeepMind<br>
**Released:** 2026-03-31

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Gemma 4** is a suite of related architectures rather than one uniform network. E2B and E4B target mobile and edge deployment; 31B is a dense server-class decoder; and 26B-A4B is a sparse Mixture-of-Experts model with roughly 4B active parameters. These branches accept images through improved visual processing, while the mobile models additionally support audio. The distinctive **Gemma 4 12B Unified** removes separate vision and audio encoders: raw image patches and audio features enter through direct linear projections, reducing duplicated encoder memory and latency. Context windows reach 128K or 256K depending on the variant.

Pretraining jointly develops language, code, reasoning, and multimodal understanding, followed by instruction tuning, safety alignment, and reasoning post-training. A configurable thinking mode lets the instruction-tuned models emit a reasoning trace before answering. The family also ships dedicated multi-token-prediction draft models for speculative decoding, while per-layer embeddings and shared key-value caching improve inference efficiency in applicable variants.

Google describes a large multilingual mixture containing web text, documents, code, mathematics, image-text pairs, OCR and document examples, audio, and video-derived supervision. The complete record-level training corpus is not released, so benchmark collections should not be mistaken for an exhaustive dataset list. Image support includes variable resolutions and aspect ratios; native audio and video coverage depends on the specific branch.

</details>

### **Penguin-VL: Efficient VLMs with LLM-Based Vision Encoders**

Penguin-VL challenges the assumption that compact VLMs require contrastively pretrained vision backbones, adapting a text-only language model into a fine-grained visual encoder for efficient image and video understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2603.06569-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.06569) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/tencent-ailab/Penguin-VL) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/tencent/penguin-vl)

Boqiang Zhang, Lei Ke, Ruihan Yang, Qi Gao, Tianyuan Qu, Rossell Chen, Dong Yu, Leoweiliang<br>
**Released:** 2026-03-06

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Penguin-VL** builds its Penguin-Encoder by initializing a vision backbone from a text-only Qwen3-0.6B model. Causal attention is converted to bidirectional attention, and 2D rotary position embeddings permit variable-resolution visual tokenization. The encoder is connected to compact language decoders to form 2B- and 8B-class VLMs. For video, **Temporal Redundancy-Aware compression** assigns larger token budgets to informative key frames and compresses redundant intermediate frames, allowing longer sequences within a fixed visual-token budget.

Training begins with mixed-supervision encoder adaptation. A teacher vision encoder supplies amplitude, direction, and relational distillation targets while image reconstruction-style supervision stabilizes the LLM-to-vision conversion. A low-to-high-resolution curriculum then aligns the encoder for fine-grained perception, followed by unified image and video multimodal pretraining and instruction tuning. This recipe is designed to preserve OCR, spatial, temporal, and dense-captioning signals that category-oriented contrastive objectives can suppress.

The project releases Penguin-Recap-I, a reconstructed high-quality image training set, and Penguin-Recap-V, video annotations at dense timestamp, paragraph, and whole-video granularities. These resources complement broader image-text, document, OCR, mathematical reasoning, and video instruction mixtures described in the report. The authors' comparisons attribute the compact models' gains primarily to visual representation quality and token allocation rather than parameter scaling alone.

</details>

### **Phi-4-Reasoning-Vision: Compact Multimodal Reasoning**

Phi-4-Reasoning-Vision-15B is a compact open-weight model that combines high-resolution perception with selectable direct and chain-of-thought modes, focusing on mathematical, scientific, document, and GUI reasoning.

[![arXiv](https://img.shields.io/badge/arXiv-2603.03975-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.03975) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/Phi-4-reasoning-vision-15B) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B)

Jyoti Aneja, Michael Harrison, Neel Joshi, Tyler LaBonte, John Langford, Eduardo Salinas<br>
**Released:** 2026-03-04

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Phi-4-Reasoning-Vision-15B** uses a mid-fusion architecture built from the Phi-4-Reasoning language backbone and a SigLIP 2 vision encoder. Visual features are projected into the language embedding space and inserted into the decoder sequence. Dynamic-resolution processing produces up to 3,600 visual tokens, supporting fine-grained documents and interface screenshots. Bidirectional attention is restricted to tokens within each image, improving spatial interaction while avoiding the overfitting Microsoft observed with broader bidirectional attention.

The model is trained through supervised fine-tuning on a deliberately mixed curriculum of reasoning and non-reasoning examples. Explicit `<think>` and `<nothink>` modes allow one checkpoint to use extended reasoning for mathematics and science or answer directly for perception-heavy tasks such as captioning, OCR, detection, and grounding. Microsoft reports four days of training on 240 NVIDIA B200 GPUs, followed by safety SFT and automated red-team evaluation.

Training data primarily comprises heavily filtered and corrected open-source vision-language datasets, augmented with synthetic examples, internally produced domain data, and targeted acquisitions. Microsoft emphasizes filtering, error correction, and synthetic augmentation as larger contributors than raw scale, but does not publish a complete source-level dataset inventory. Public evaluations span scientific diagrams, charts, mathematical reasoning, OCR, and GUI grounding; those benchmarks are evaluation references, not a claimed exhaustive training list.

</details>

### **V-SONAR and V-LCM: Vision-Language Modeling in Concept Space**

V-SONAR aligns image and video representations with SONAR's multilingual concept space, enabling text-trained concept models to process vision. V-LCM then instruction-tunes a latent-diffusion Large Concept Model directly over unified visual and linguistic embeddings.

[![arXiv](https://img.shields.io/badge/arXiv-2603.01096-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.01096) [![OpenReview](https://img.shields.io/badge/ICLR-2026-8c1b13?style=flat-square)](https://openreview.net/forum?id=4LiX5ddGcU)

Yifu Qiu, Paul-Ambroise Duquenne, Holger Schwenk<br>
**Released:** 2026-03-01

<details>
<summary>ℹ️ <i>More Information</i></summary>

**V-SONAR** starts with Meta's Perception Encoder for image and video features and learns a lightweight projector into the pre-existing SONAR embedding space. Positional embeddings and a temporal-attention layer preserve frame order, while attention pooling produces a modality-level representation compatible with SONAR text concepts. Because the target space is shared with OMNISONAR, the representation can be decoded into text across SONAR's multilingual coverage. A text-only Large Concept Model can also consume these aligned visual embeddings without vision-specific pretraining.

V-SONAR is trained with a coarse-to-fine post-hoc alignment curriculum rather than retraining the foundation encoders from scratch. The alignment uses paired visual-caption supervision and SONAR text embeddings, with staged optimization and asynchronous learning rates for the newly initialized projector and pretrained encoder. **V-LCM** then concatenates V-SONAR visual concepts with SONAR language concepts and retains LCM's two-tower contextualizer and denoiser architecture. It predicts the next continuous concept embedding with the same latent-diffusion objective used during text-only LCM pretraining.

Vision-language instruction tuning uses M3IT, which spans image and video inputs, eight task categories, and 80 languages. Retrieval and captioning experiments additionally use VATEX, DREAM-1K, PE-Video, and VideoXum. The resulting system separates perception, multilingual semantic alignment, and generative concept modeling instead of converting every modality into a single discrete token vocabulary.

</details>

### **Qwen3.5: Native Multimodal Hybrid-Attention Models**

Qwen3.5 unifies text, image, and video understanding in one model family while combining hybrid linear attention with sparse expert routing. Qwen3.6 continues the same native-multimodal direction with stronger agentic coding and updated dense and MoE checkpoints.

[![Blog](https://img.shields.io/badge/Official-Qwen3.5-b31b1b?style=flat-square)](https://qwen.ai/blog?id=qwen3.5) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen3.6) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)

Qwen Team<br>
**Released:** 2026-02-16

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Qwen3.5** is a natively multimodal decoder family rather than a separate text model with a later VL edition. Its flagship 397B-A17B checkpoint uses sparse Mixture-of-Experts routing, activating about 17B parameters per token, and combines **Gated DeltaNet** linear-attention layers with standard softmax-attention layers. Visual inputs are handled inside the same model interface as text, supporting image and video reasoning, tool use, coding, and long-context agent workflows. Smaller dense and MoE variants extend the design across deployment scales.

The training pipeline combines large-scale multimodal pretraining with supervised fine-tuning and reinforcement learning. Qwen describes a scalable asynchronous RL system spanning text, multimodal, and multi-turn interaction, enabling a single checkpoint to switch between thinking and non-thinking behavior. The data mixture covers natural language, code, mathematics, images, video, documents, GUI interactions, tool calls, and multi-turn agent trajectories; the full source-level corpus is not publicly enumerated.

**Qwen3.6** is best treated as a continuation of this architecture rather than a separate VLM family. Its open dense and sparse checkpoints retain native multimodality and unified thinking and non-thinking modes while prioritizing stability and agentic coding. Folding these releases into the Qwen3.5 entry keeps the catalog focused on architectural changes rather than model-version churn.

</details>

### **Youtu-VL: Unified Autoregressive Supervision for Dense Vision**

Youtu-VL extends the language vocabulary with learned visual codes, making visual tokens prediction targets so one autoregressive model can emit segmentation, depth, pose, and detection without task-specific heads.

[![arXiv](https://img.shields.io/badge/arXiv-2601.19798-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.19798) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/TencentCloudADP/youtu-vl) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/tencent/Youtu-VL-4B-Instruct)

Tencent Youtu Lab<br>
**Released:** 2026-01-27

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Vision-Language Unified Autoregressive Supervision** learns a visual codebook whose symbols expand the text vocabulary into one multimodal vocabulary. Visual tokens are not passive conditioning embeddings: the decoder predicts them alongside text through joint visual-token and text reconstruction.

Because dense outputs are serialized in the same vocabulary, a standard autoregressive VLM can produce segmentation, monocular depth, human pose, and object detections without separate heads. This turns dense perception into native generation while retaining ordinary visual question answering and instruction following.

</details>

### **Kimi K2.5 and K2.6: Native Multimodal Agentic MoE**

Kimi K2.5 is a trillion-parameter native multimodal MoE for reasoning, coding, and computer use; K2.6 keeps the topology while extending context and agentic post-training.

[![arXiv](https://img.shields.io/badge/arXiv-2602.02276-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.02276) [![HuggingFace](https://img.shields.io/badge/HuggingFace-K2.6-blue?style=flat-square)](https://huggingface.co/moonshotai/Kimi-K2.6) [![Release](https://img.shields.io/badge/Official-K2.6-blue?style=flat-square)](https://www.kimi.com/blog/kimi-k2-6)

Kimi Team, Moonshot AI<br>
**Released:** 2026-01-27

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Kimi K2.5** connects MoonViT to a 1T-total, 32B-active decoder with 61 layers, Multi-head Latent Attention, and 384 routed experts. Eight routed experts plus shared capacity activate per token. Native multimodal training supports images, video, reasoning, code, tools, and GUI interaction within the same model.

Released April 20, **Kimi K2.6** retains the K2.5 architecture while extending context to 256K and strengthening coding, research, and multimodal-agent post-training. It is folded here rather than receiving a duplicate timeline row; Kimi K3 is separate because its attention, residual, sparsity, scale, and vision stack change materially.

</details>

### **Step3-VL-10B: Language-Aligned Perception with 16× Token Compression**

Step3-VL-10B combines a 1.8B language-aligned perception encoder with a Qwen3-8B decoder and aggressively compresses high-resolution visual features before language reasoning.

[![arXiv](https://img.shields.io/badge/arXiv-2601.09668-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.09668) [![Project](https://img.shields.io/badge/Project-Step3--VL--10B-blue?style=flat-square)](https://stepfun-ai.github.io/Step3-VL-10B/)

StepFun<br>
**Released:** 2026-01-14

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Step3-VL-10B** is architecturally distinct from the much larger 2025 Step3 sparse-MoE system. Its 1.8B perception encoder is aligned to language during pretraining, then a projector with two stride-2 stages reduces the spatial sequence by 16× before feeding a Qwen3-8B decoder.

Images combine one 728×728 global view with 504×504 local crops, treated as independent batch items. Explicit newline tokens mark patch rows while standard one-dimensional RoPE is retained. The PaCoRe parallel proposer-and-synthesizer reasoning method is a post-training and inference addition rather than the defining visual architecture.

</details>

### **ERNIE 5.0: Unified Autoregressive Omnimodal Mixture of Experts**

ERNIE 5.0 unifies text, image, video, and audio understanding and generation in a 2.4T ultra-sparse MoE trained through modality-specific forms of grouped next-token prediction.

[![arXiv](https://img.shields.io/badge/arXiv-2602.04705-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.04705) [![Architecture](https://img.shields.io/badge/Official-Architecture-blue?style=flat-square)](https://ernie.baidu.com/blog/posts/ernie5.0/)

Baidu ERNIE Team<br>
**Released:** 2025-11-13

<details>
<summary>ℹ️ <i>More Information</i></summary>

**ERNIE 5.0** maps language, images, video, and audio into one autoregressive framework using Next-Group-of-Tokens Prediction. Text uses next-token and multi-token prediction; visual generation predicts the next frame and scale; audio generation predicts codec tokens depth-wise. Modality-agnostic expert routing activates less than 3% of the 2.4T-parameter network.

Elastic depth, width, and top-k sparsity train a super-network from which smaller deployment configurations can be extracted. Baidu first unveiled ERNIE 5.0 at Baidu World on November 13, 2025, formally released it in January 2026, and published the technical report on February 4; the timeline therefore uses the earliest documented public disclosure.

</details>

### **DeepSeek-OCR: Visual Context Compression through DeepEncoder**

DeepSeek-OCR treats document vision as a context-compression mechanism, representing thousands of text tokens with a much smaller sequence of visual tokens before decoding them with a sparse language model.

[![arXiv](https://img.shields.io/badge/arXiv-2510.18234-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2510.18234) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/deepseek-ai/DeepSeek-OCR) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

DeepSeek-AI<br>
**Released:** 2025-10-20

<details>
<summary>ℹ️ <i>More Information</i></summary>

**DeepSeek-OCR** consists of **DeepEncoder** and a DeepSeek-3B-MoE decoder with roughly 570M active parameters. DeepEncoder combines a SAM-derived local-perception pathway, a CLIP-derived semantic pathway, convolutional compression, and windowed and global attention to turn high-resolution pages into as few as 64 to 800 visual tokens. Multiple resolution modes trade recognition fidelity for compression.

Training first develops the encoder's visual-text representation and then jointly optimizes document decoding with the MoE language model. Data covers multilingual text pages, natural images, formulas, tables, charts, diagrams, and structured document conversion, supplemented with synthetic renderings. The work frames OCR as an experiment in optical compression for future long-context memory, rather than only a document benchmark model.

**DeepSeek-OCR 2**, released January 27, 2026, replaces the fixed raster-scan assumption with **DeepEncoder V2** and Visual Causal Flow. The encoder dynamically reorders visual tokens according to document semantics before language decoding, exploring whether cascaded one-dimensional causal reasoning can better represent complex two-dimensional layouts. [Paper](https://arxiv.org/abs/2601.20552) · [Repository](https://github.com/deepseek-ai/DeepSeek-OCR-2)

</details>

### **PaddleOCR-VL: Ultra-Compact Multilingual Document Parsing**

PaddleOCR-VL combines a NaViT-style dynamic-resolution encoder with ERNIE-4.5-0.3B to parse multilingual documents using only 0.9B parameters.

[![arXiv](https://img.shields.io/badge/arXiv-2510.14528-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2510.14528) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/PaddlePaddle/PaddleOCR) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

PaddleOCR Team, Baidu<br>
**Released:** 2025-10-16

<details>
<summary>ℹ️ <i>More Information</i></summary>

**PaddleOCR-VL-0.9B** connects a NaViT-style dynamic-resolution visual encoder to the compact ERNIE-4.5-0.3B language model. Variable-resolution packing preserves page detail and aspect ratio without forcing every document through a fixed square canvas. The decoder emits text and structured representations for paragraphs, tables, formulas, charts, and other document elements across 109 languages.

Its curriculum combines element-level recognition with page-level document parsing, followed by instruction tuning on complex layouts and structured outputs. Training data includes multilingual OCR, synthetic and scanned documents, tables, mathematical expressions, charts, reading-order annotations, and layout-rich pages. **PaddleOCR-VL-1.5** (January 2026) adds robust physical-document distortions, seal recognition, and text spotting; **1.6** (June 2026) adds region-aware data optimization and progressive reinforcement-learning post-training without changing the core architecture.

</details>

### **Qwen3-VL: DeepStack Vision-Language Models**

Qwen3-VL expands the Qwen vision-language family with dense and sparse-MoE checkpoints, native long multimodal context, and stronger spatial, video, OCR, visual-agent, and visual-coding capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2511.21631-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2511.21631) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen3-VL) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/Qwen/qwen3-vl)

Qwen Team<br>
**Released:** 2025-09-22

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Qwen3-VL** combines a native-dynamic-resolution Vision Transformer with Qwen3 language backbones in dense 2B, 4B, 8B, and 32B configurations and sparse 30B-A3B and 235B-A22B Mixture-of-Experts configurations. Its **DeepStack** mechanism injects features from multiple ViT depths into corresponding language-model layers instead of relying only on the visual encoder's final layer. **Interleaved-MRoPE** distributes temporal, height, and width position information across rotary-embedding frequencies, while explicit text-timestamp alignment improves event localization in video. The family supports interleaved text, images, and video in a native 256K-token context, with documented extrapolation for longer contexts.

Training proceeds through large-scale multimodal pretraining followed by supervised instruction tuning and reasoning-oriented post-training. The released Instruct checkpoints target direct response and agent interaction, whereas Thinking checkpoints generate extended reasoning for visual mathematics, spatial analysis, and other difficult multimodal tasks. The curriculum emphasizes recognition, multilingual OCR, document structure, 2D and 3D grounding, long-video understanding, GUI operation, and code generation from visual inputs.

The technical report describes a broad mixture of text, image-text, document, OCR, grounding, chart, multi-image, video, and agent-interaction data. Qwen does not publish a reproducible itemized list of every pretraining source, so individual benchmark datasets should not be presented as the complete training corpus.

</details>

### **Step3: Model-System Co-Design for Cost-Effective Multimodal Intelligence**

Step3 is a 321B-total, 38B-active multimodal MoE whose architecture and serving system are co-designed to reduce the communication cost of sparse expert decoding.

[![arXiv](https://img.shields.io/badge/arXiv-2507.19427-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2507.19427) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/stepfun-ai/Step3) [![Website](https://img.shields.io/badge/Website-Step3-blue?style=flat-square)](https://chat.stepfun.com/research/en/step3)

StepFun<br>
**Released:** 2025-07-25

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Step3** uses a 321B-parameter sparse-MoE decoder with approximately 38B parameters active per token. The language stack contains dense lower layers followed by routed expert layers, while its vision pathway supports images and video for multimodal reasoning. The model is designed together with an **Attention-FFN Disaggregation** serving architecture, separating attention and expert computation so hardware placement and communication topology match their different workloads.

Training combines large-scale text and multimodal pretraining with instruction tuning and reasoning-oriented reinforcement learning. The data mixture spans language, code, mathematics, image-text, OCR, documents, charts, multi-image, video, and multimodal reasoning. The system report discloses architecture and deployment mechanisms more clearly than the source-level training corpus, which remains categorically described.

</details>

### **GLM-4.1V-Thinking: General-Purpose Multimodal Reasoning through Curriculum-Sampled RL**

GLM-4.1V-9B-Thinking combines a native-resolution AIMv2-based vision stack with long-chain-of-thought post-training and Reinforcement Learning with Curriculum Sampling to strengthen reasoning across visual, document, video, grounding, and agent tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2507.01006-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2507.01006) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/zai-org/GLM-V) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)

GLM-V Team, Zhipu AI and Tsinghua University<br>
**Released:** 2025-07-01

<details>
<summary>ℹ️ <i>More Information</i></summary>

**GLM-4.1V-9B-Thinking** joins an AIMv2-Huge-initialized vision encoder, an MLP projector, and the GLM-4-9B-0414 language decoder. The vision tower replaces two-dimensional patch convolutions with 3D convolutions, temporally downsampling video while duplicating single images for consistent processing. Interpolated absolute position embeddings and 2D RoPE support native resolutions and extreme aspect ratios; the language decoder applies 3D RoPE to multimodal tokens. Explicit timestamp tokens after video frames supply temporal position and distance cues.

Training begins with broad multimodal pretraining, followed by continual training on video, higher-resolution imagery, and sequences up to 32K tokens. Supervised fine-tuning then teaches a standardized long-reasoning format with separate `<think>` and `<answer>` spans. The final stage introduces **Reinforcement Learning with Curriculum Sampling (RLCS)**. Rather than sampling a fixed task mixture, RLCS estimates changing model competence and prioritizes informative problems of suitable difficulty. Domain-specific rule, model, and hybrid rewards cover both verifiable and open-ended tasks.

Pretraining draws on curated caption pairs, knowledge-rich web and academic-book interleaving, 220M OCR images, natural-image and GUI grounding, instructional video, and text replay. SFT and RL span STEM reasoning, charts, documents, video, grounding, coding, GUI agents, and general visual instruction following. The later GLM-4.5V and GLM-4.6V families scale and extend this framework; they are successors, not separate architectures in this catalog.

</details>

### **ERNIE 4.5-VL: Heterogeneous Modality Mixture-of-Experts**

ERNIE 4.5-VL combines shared and modality-specific experts with isolated routing and balancing losses so visual learning can reinforce rather than degrade language capability.

[![Website](https://img.shields.io/badge/Report-ERNIE_4.5-blue?style=flat-square)](https://ernie.baidu.com/blog/posts/ernie4.5/) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/PaddlePaddle/ERNIE) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/baidu/ernie-45)

Baidu ERNIE Team<br>
**Released:** 2025-06-30

<details>
<summary>ℹ️ <i>More Information</i></summary>

**ERNIE 4.5-VL** includes 424B-total/47B-active and 28B-total/3B-active multimodal MoE models, plus dense members of the broader ERNIE 4.5 family. Its **heterogeneous modality MoE** shares some experts across text and visual tokens while reserving other experts for individual modalities. Modality-isolated routing, router orthogonality loss, and multimodal token-balancing loss reduce competition between modalities.

The models are jointly pretrained on text, images, and video, then post-trained in both direct-response and thinking modes. PaddlePaddle infrastructure uses heterogeneous hybrid parallelism, hierarchical load balancing, FP8 mixed precision, and fine-grained recomputation to train and serve the sparse models efficiently. Baidu describes broad text, visual-knowledge, OCR, document, chart, video, and reasoning mixtures but does not provide a complete dataset manifest.

</details>

### **MiMo-VL: Multimodal Pretraining with Mixed On-Policy Reinforcement Learning**

MiMo-VL combines a four-stage, 2.4T-token multimodal curriculum with mixed on-policy reinforcement learning across perception, reasoning, and GUI grounding.

[![arXiv](https://img.shields.io/badge/arXiv-2506.03569-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2506.03569) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/XiaomiMiMo/MiMo-VL) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/XiaomiMiMo/mimo-vl)

Xiaomi MiMo Team<br>
**Released:** 2025-06-04

<details>
<summary>ℹ️ <i>More Information</i></summary>

**MiMo-VL-7B** joins a high-resolution vision encoder and projector to a 7B language decoder and is released as supervised and reinforcement-learned checkpoints. Its primary contribution is the training system rather than an exotic connector: long-chain-of-thought examples are introduced during pretraining, not only during post-training, so multimodal reasoning is developed alongside perception and language.

Four pretraining stages consume approximately 2.4T tokens and progressively align vision, general multimodal knowledge, high-quality reasoning, and long-context capabilities. **Mixed On-Policy Reinforcement Learning** then trains one checkpoint across general QA, mathematics, grounding, and GUI tasks using domain-appropriate rule and model rewards. The mixture includes text replay, image-text, OCR, document, chart, video, spatial-grounding, GUI, and reasoning data; a complete source-level inventory is not published.

</details>

### **BAGEL: A Mixture-of-Transformer-Experts for Unified Understanding and Generation**

BAGEL unifies visual understanding, autoregressive language modeling, and rectified-flow image generation in a decoder-only Mixture-of-Transformer-Experts whose modality-specific parameters communicate through shared self-attention.

[![arXiv](https://img.shields.io/badge/arXiv-2505.14683-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2505.14683) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/ByteDance-Seed/Bagel) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, Haoqi Fan<br>
**Released:** 2025-05-20

<details>
<summary>ℹ️ <i>More Information</i></summary>

**BAGEL** is a unified decoder-only model with 14B total and 7B active parameters. Its **Mixture-of-Transformer-Experts** architecture duplicates the Qwen2.5-initialized transformer into understanding and generation experts: text and SigLIP 2 ViT tokens use the understanding parameters, while FLUX-VAE latent tokens use the generation parameters. Both experts operate on one interleaved sequence through shared self-attention at every layer, avoiding a small connector bottleneck between comprehension and synthesis. Text is learned with next-token prediction; images are generated with rectified flow.

A generalized causal-attention scheme lets later text or images attend to earlier clean visual representations while preventing access to noised generation targets. Visual understanding uses a native-aspect-ratio-adapted SigLIP2-So400m encoder and MLP connector; generation uses a frozen FLUX VAE. Training progresses through alignment, large-scale pretraining, continued training, and supervised fine-tuning, jointly optimizing cross-entropy and flow-matching objectives.

BAGEL is pretrained on trillions of tokens assembled from text-only, paired image-text, and interleaved multimodal sources. The reported inventory includes 400M text samples, 500M understanding pairs, 1.6B generation pairs, 100M interleaved understanding samples, 45M video-derived sequences, and 20M web documents. OCR, charts, grounding, editing sets, and 500K reasoning-augmented generation and manipulation examples support document reading, spatial control, image editing, and long-context multimodal reasoning.

</details>

### **Seed1.5-VL: Sparse-MoE Multimodal Understanding and Agentic Reasoning**

Seed1.5-VL couples a 532M-parameter vision encoder with a 20B-active sparse-MoE decoder for image, video, 2D and 3D grounding, GUI interaction, and long-chain-of-thought reasoning.

[![arXiv](https://img.shields.io/badge/arXiv-2505.07062-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2505.07062) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/ByteDance-Seed/Seed1.5-VL)

ByteDance Seed Team<br>
**Released:** 2025-05-11

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Seed1.5-VL** uses a compact vision tower and a large sparse-MoE language model, maintaining 20B active decoder parameters while scaling total capacity substantially higher. Its visual pathway supports native-aspect-ratio images, multiple images, and video, with explicit support for spatial grounding and GUI control. The same model can provide short direct answers or extended multimodal reasoning.

Training progresses through vision-language pretraining, instruction tuning, long-chain-of-thought cold start, and reinforcement learning. The report emphasizes careful balancing of general text, visual knowledge, OCR and documents, charts, grounding, video, 3D understanding, GUI trajectories, games, and verifiable multimodal-reasoning problems. Source-level corpus provenance and all mixture proportions are not publicly reproducible. The arXiv v1 date is used because an earlier generally accessible official launch date is not unambiguously documented.

</details>

### **InternVL3 and InternVL3.5: Native Multimodal Pretraining and Adaptive Resolution**

InternVL3 moves the InternVL family to native multimodal pretraining, while InternVL3.5 adds coarse-to-fine reinforcement learning and dynamically routed visual resolution.

[![arXiv](https://img.shields.io/badge/arXiv-2504.10479-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.10479) [![arXiv](https://img.shields.io/badge/arXiv-2508.18265-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.18265) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenGVLab/InternVL) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/OpenGVLab/internvl35)

InternVL Team, OpenGVLab<br>
**Released:** 2025-04-11

<details>
<summary>ℹ️ <i>More Information</i></summary>

**InternVL3** retains the InternViT-MLP-LLM structure and dynamic image tiling of earlier InternVL releases, but jointly learns language and multimodal knowledge during native pretraining instead of treating vision alignment primarily as a later adaptation stage. The family spans dense and sparse-MoE language backbones. Its training recipe combines multimodal pretraining, supervised fine-tuning, Mixed Preference Optimization, and test-time scaling for visual reasoning.

Released on August 26, 2025, **InternVL3.5** is part of the same evolving family. It introduces **Cascade RL**, using offline RL for stable coarse alignment followed by online RL for refinement. A **Visual Resolution Router** selects visual-token resolution according to input complexity, while **Decoupled Vision-Language Deployment** places the vision encoder and language model on different devices to balance inference load. Training covers image, multi-image, document, chart, video, GUI, grounding, multilingual, and reasoning data; full pretraining provenance is not enumerated.

</details>

### **Kimi-VL: Native-Resolution Vision with a Sparse MoE Decoder**

Kimi-VL couples the native-resolution MoonViT encoder to a sparse Mixture-of-Experts language decoder, activating 2.8B of 16B decoder parameters while supporting images, video, agent interaction, and 128K-token contexts.

[![arXiv](https://img.shields.io/badge/arXiv-2504.07491-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2504.07491) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/MoonshotAI/Kimi-VL) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)

Kimi Team<br>
**Released:** 2025-04-10

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Kimi-VL-A3B** consists of MoonViT, a two-layer MLP projector, and a Moonlight-derived MoE language model with 16B total and 2.8B activated decoder parameters. MoonViT packs variable-sized patch sequences without tiling images into separately processed crops. It combines interpolated SigLIP-SO400M position embeddings with two-dimensional RoPE, preserving pretrained visual knowledge while representing fine spatial positions. A 2×2 pixel-shuffle operation compresses its output before projection into the language model.

After the language checkpoint's 5.2T-token text-only phase, Kimi-VL undergoes standalone vision training, vision-language alignment, joint pretraining, a high-quality cooldown, and long-context activation. These stages consume 4.4T additional tokens, including 2T for MoonViT training, 0.1T for alignment, and 2.3T across joint stages. Context is progressively extended from 8K to 128K. The instruct model receives joint supervised fine-tuning. Kimi-VL-Thinking adds long-chain-of-thought SFT and reinforcement learning; the later Thinking-2506 checkpoint is folded into this family.

The multimodal mixture is organized into six categories: caption, interleaved image-text, OCR, knowledge, video, and agent data. MoonViT learns from alt text, synthetic captions, grounding boxes, and OCR targets using both SigLIP-style contrastive and captioning losses. Long-video, long-document, academic visual QA, GUI-grounding, and text replay data provide long-context and agent capabilities while preserving language performance.

</details>

### **Llama 4 Scout and Maverick: Native Multimodal Mixture-of-Experts Models**

Llama 4 introduces early-fusion multimodality to the Llama family through sparse Mixture-of-Experts decoders with 17B active parameters and contexts extending from one million to ten million tokens.

[![Model Card](https://img.shields.io/badge/Meta-Model_Card-blue?style=flat-square)](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/meta-llama/llama-4)

Meta<br>
**Released:** 2025-04-05

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Llama 4 Scout** and **Llama 4 Maverick** are natively multimodal, autoregressive MoE models rather than text decoders retrofitted only during instruction tuning. Scout has 109B total parameters across 16 experts and Maverick approximately 400B across 128 experts; both activate about 17B parameters per token. Early fusion places image and multilingual text tokens in a shared model sequence. Scout supports a 10M-token context, whereas Maverick supports 1M tokens.

Scout was trained on roughly 40T tokens and Maverick on roughly 22T. Meta describes a mixture of public data, licensed data, information from its products and services, and interactions with Meta AI, without releasing a complete corpus manifest. Post-training combines supervised fine-tuning, online reinforcement learning, and direct preference optimization. Maverick was co-distilled from the unreleased Llama 4 Behemoth teacher; Scout also uses distillation during training.

</details>

### **Qwen2.5-Omni: Streaming Multimodal Perception and Speech Generation**

Qwen2.5-Omni understands text, images, audio, and video while generating text and natural speech through a streaming Thinker-Talker architecture.

[![arXiv](https://img.shields.io/badge/arXiv-2503.20215-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2503.20215) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen2.5-Omni) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

Qwen Team<br>
**Released:** 2025-03-26

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Qwen2.5-Omni** is an end-to-end any-input model whose block-wise audio and vision encoders accept streaming speech, images, and video alongside text. **Time-aligned Multimodal RoPE (TMRoPE)** interleaves audio and video representations on a synchronized timeline. Its **Thinker-Talker** design separates semantic reasoning from speech realization: the Thinker produces text and hidden representations, while a dual-track autoregressive Talker conditions on those states to generate speech tokens concurrently.

A sliding-window diffusion transformer converts audio tokens into waveform output without waiting for a complete response, reducing first-packet latency. Training jointly aligns text, vision, video, audio, and speech-generation objectives, followed by multimodal instruction tuning. The report describes caption, OCR, video, audio-transcription, speech, and general text mixtures but does not publish a reproducible source-level inventory.

</details>

### **Gemma 3: Long-Context Multimodality with Efficient Interleaved Attention**

Gemma 3 combines a frozen SigLIP vision encoder with a decoder-only language model whose five-local-to-one-global attention pattern reduces long-context KV-cache cost while retaining 128K-token multimodal context.

[![arXiv](https://img.shields.io/badge/arXiv-2503.19786-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2503.19786) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-deepmind/gemma) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/google/gemma-3-release)

Gemma Team<br>
**Released:** 2025-03-12

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Gemma 3** is a family of 1B, 4B, 12B, and 27B open-weight models; the 4B, 12B, and 27B variants accept images as well as text. A shared, frozen 400M-parameter SigLIP vision encoder processes images at 896×896 resolution. Its embeddings are average-pooled into a fixed sequence of 256 soft visual tokens, while a Pan-and-Scan preprocessing method handles non-square and high-resolution inputs through crops. The 1B model remains text-only.

For efficient long-context inference, Gemma 3 changes the decoder from Gemma 2's balanced attention schedule to **five local sliding-window layers for every global-attention layer**. Local attention uses a 1,024-token window, substantially limiting KV-cache growth; global layers retain access to the complete context. The larger models are pretrained at 32K context and extended to 128K near the end of pretraining through RoPE rescaling.

Training relies heavily on knowledge distillation from a larger teacher, updating the language model while keeping the vision encoder frozen. Post-training combines supervised fine-tuning with preference-based reinforcement learning. The pretraining mixture contains text, code, mathematics, multilingual material, and multimodal examples; the report states token totals of 2T, 4T, 12T, and 14T for the four model sizes but does not publish a complete dataset inventory.

</details>

### **Aya Vision: Multilingual Multimodality through Cross-Modal Model Merging**

Aya Vision uses synthetic multilingual visual instruction data and cross-modal model merging to add image understanding while preserving the language capabilities of Aya Expanse.

[![arXiv](https://img.shields.io/badge/arXiv-2505.08751-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2505.08751) [![Website](https://img.shields.io/badge/Website-Aya_Vision-blue?style=flat-square)](https://cohere.com/blog/aya-vision) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/CohereLabs/aya-vision-8b)

Cohere Labs<br>
**Released:** 2025-03-04

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Aya Vision** is released in 8B and 32B configurations supporting 23 languages. It connects a vision encoder to Aya Expanse-derived language models and addresses the loss of text-only capability that often occurs when vision is introduced. Its principal modeling contribution is **cross-modal model merging**, which combines separately learned text and multimodal strengths rather than relying only on sequential fine-tuning.

Training uses a synthetic annotation pipeline that generates and filters culturally and linguistically diverse multimodal instructions, avoiding dependence on literal machine translation alone. The mixture covers multilingual visual question answering, captions, OCR, image-grounded generation, and text-and-image translation. The full pretraining corpus is not released; the paper's reproducible contributions center on its annotation, merging, and evaluation recipes.

</details>

### **Phi-4-multimodal: Text, Vision, and Speech through Mixture-of-LoRAs**

Phi-4-multimodal extends a compact language backbone to images and audio using modality encoders, projectors, LoRA adapters, and modality-specific routing, allowing several input combinations without destructive interference between learned capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2503.01743-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2503.01743) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

Microsoft Phi Team<br>
**Released:** 2025-03-03

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Phi-4-multimodal** is a 5.6-billion-parameter multimodal model built around the Phi-4-Mini language model. Separate vision and speech encoders transform non-text inputs, while modality projectors map their representations into the language model's embedding space. Its defining mechanism is a **Mixture-of-LoRAs**: modality-specific LoRA parameters and routers adapt the shared decoder for vision-language, speech-language, and joint vision-speech inference. The design isolates modality extensions more effectively than fully merging every adaptation into the base weights; the speech and audio LoRA component itself contains 460 million parameters.

The underlying language model uses grouped-query attention, a 200K-token vocabulary, and supports a 128K-token context. Multimodal training first aligns the modality encoders and projectors, then jointly optimizes modality-specific adaptations. The released instruct checkpoint receives supervised fine-tuning followed by direct preference optimization and reinforcement learning from human feedback for instruction following and safety.

Its data recipe builds on the language, vision, and speech research used for Phi-3.5 and Phi-4. Vision supervision covers captioning, OCR, charts, diagrams, document understanding, visual question answering, and multi-image reasoning. Speech data spans automatic speech recognition, speech translation, and summarization across multiple languages and acoustic conditions. The model card provides a data summary rather than a complete sample-level corpus inventory, so undisclosed sources should not be inferred from benchmark coverage.

</details>

### **SigLIP 2: Multilingual Vision-Language Encoders with Native-Aspect-Ratio Support**

SigLIP 2 extends sigmoid-loss image-text pretraining into a multilingual, localization-aware representation learner, with NaFlex checkpoints that preserve native aspect ratios and support multiple token budgets from one model.

[![arXiv](https://img.shields.io/badge/arXiv-2502.14786-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.14786) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/google/siglip2)

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, et al.<br>
**Released:** 2025-02-20

<details>
<summary>ℹ️ <i>More Information</i></summary>

**SigLIP 2** retains the dual-encoder Vision Transformer design and pairwise sigmoid image-text objective of SigLIP, but broadens the representation learned by the vision tower. Its unified recipe combines contrastive image-text learning with captioning-based pretraining, masked prediction, self-distillation, and online data curation. These additions improve not only zero-shot classification and retrieval, but also localization and dense tasks such as segmentation, depth estimation, and surface-normal prediction. Models are released at ViT-B, ViT-L, So400m, and ViT-g scales.

Alongside conventional fixed-resolution checkpoints, the **NaFlex** variants use patch-and-pack processing to retain an image's native aspect ratio and accept different sequence lengths with a single checkpoint. This avoids distortive square resizing and lets users trade visual-token count for inference cost, which is especially useful for documents, screenshots, and OCR-heavy inputs. Standard fixed-resolution models additionally receive a self-distillation stage.

Training uses **WebLI**, comprising 10 billion images and 12 billion alt-texts across 109 languages. The reported mixture draws 90% of image-text pairs from English pages and 10% from non-English pages, while debiasing filters target representation and association biases involving sensitive attributes. SigLIP 2 is positioned both as a standalone multilingual image-text encoder and as a stronger frozen vision tower for downstream VLMs.

</details>

### **ShowUI: Vision-Language-Action Modeling for GUI Agents**

ShowUI is a lightweight 2B vision-language-action model built on Qwen2-VL for screenshot grounding and GUI navigation. It reduces redundant screenshot tokens through UI-guided selection and unifies grounding and navigation examples as interleaved vision-language-action streams.

[![arXiv](https://img.shields.io/badge/arXiv-2411.17465-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2411.17465) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/showlab/ShowUI) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/showlab/ShowUI-2B)

Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, Mike Zheng Shou<br>
**Released:** 2024-11-26

<details>
<summary>ℹ️ <i>More Information</i></summary>

**ShowUI** adapts Qwen2-VL-2B into an end-to-end GUI agent that predicts textual responses and executable actions from pixels. Its **UI-Guided Visual Token Selection** models screenshot patches as a connected graph, identifies visually redundant regions, and retains representative tokens during selected self-attention blocks. This exploits the repeated colors and structures common in interfaces: the paper reports removing about 33% of visual tokens during training and a 1.4× speedup without sacrificing grounding accuracy.

Training uses **Interleaved Vision-Language-Action Streaming** to express multiple GUI tasks in one causal format. For navigation, screenshots, instructions, and actions are arranged as temporal trajectories so the model can use visual-action history. For grounding, multiple query-action pairs can share a screenshot, improving data reuse and training efficiency. A resampling strategy prevents abundant task or interface categories from overwhelming smaller ones. The resulting model is trained on approximately 256K curated examples.

The instruction mixture covers web, mobile, and desktop interfaces and combines screenshot grounding with sequential navigation data. It incorporates carefully filtered public GUI sources and recaptioned annotations, then evaluates transfer on ScreenSpot-style grounding and on Mind2Web, AITW, and MiniWob navigation environments. The repository releases model weights, datasets, training and evaluation code, and the implementation of UI-guided token selection.

</details>

### **Emu3: Next-Token Prediction across Text, Image, and Video**

Emu3 tokenizes text, images, and video into discrete sequences and trains a single 8B autoregressive transformer from scratch to predict the next token. It unifies multimodal understanding and visual generation without a diffusion model, CLIP vision encoder, or separately pretrained LLM.

[![arXiv](https://img.shields.io/badge/arXiv-2409.18869-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2409.18869) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/baaivision/Emu3) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f)

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, et al.<br>
**Released:** 2024-09-27

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Emu3** first maps images and videos into discrete codes with a learned vision tokenizer, then interleaves those codes with ordinary text tokens. A single decoder-only transformer models every resulting sequence with the same causal next-token objective. Images are serialized with explicit spatial structure, while videos extend the representation across time. Because generation and perception share the same token space and transformer, Emu3 can caption and answer questions about images, generate images from text, and causally generate or continue video without adding a diffusion decoder or a compositional vision-language bridge.

The 8B transformer is pretrained from scratch on approximately 0.2 trillion visual tokens and 0.8 trillion text tokens. Pretraining establishes general multimodal generation and understanding; separate post-training paths then produce Emu3-Chat for instruction-following perception and Emu3-Gen for controllable image generation. Generation post-training includes supervised fine-tuning and preference optimization, whereas the chat model is tuned on multimodal conversations and perception tasks.

The pretraining data combines images, videos, interleaved image-text sequences, and multilingual text and code gathered and filtered at scale. Post-training uses curated image descriptions, visual question-answering and dialogue examples for Emu3-Chat, plus high-quality text-image pairs and preference data for Emu3-Gen. The project separately releases the base-stage model, chat and generation checkpoints, and the Emu3 vision tokenizer.

</details>

### **Molmo and PixMo: Open Weights, Open Data, and Grounded Pointing**

Molmo is Ai2's family of open vision-language models, while PixMo is the collection of human- and programmatically-created datasets used to train them. Molmo joins a CLIP vision encoder to OLMo, OLMoE, or Qwen2 language backbones and supports text generation, visual grounding, and pointing.

[![arXiv](https://img.shields.io/badge/arXiv-2409.17146-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2409.17146) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/allenai/molmo) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models_&_Data-blue?style=flat-square)](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, et al.<br>
**Released:** 2024-09-25

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Molmo** uses a pretrained CLIP ViT-L/14 vision encoder and a lightweight connector to supply image features to an autoregressive language model. The family spans MolmoE-1B, whose mixture-of-experts backbone has roughly 1B active and 7B total parameters, dense 7B variants based on OLMo or Qwen2, and a 72B Qwen2-based model. Multi-crop high-resolution processing improves fine-detail perception, while coordinate tokens let the model answer with points as well as natural language.

Training separates multimodal pretraining from supervised fine-tuning. Pretraining teaches image understanding primarily through dense captions, after which instruction tuning develops question answering, document reading, counting, pointing, and conversational behavior. The pipeline starts from pretrained vision and language components but releases the resulting weights, training and evaluation code, intermediate checkpoints, and data. Crucially, the team avoids distilling image understanding from proprietary VLM outputs.

**PixMo** names the data suite, not the model. PixMo-Cap contains detailed captions collected from people through speech, then transcribed and cleaned. Fine-tuning sources include PixMo-AskModelAnything, PixMo-CapQA, PixMo-Points, PixMo-Point-Explanations, PixMo-Docs, PixMo-Clocks, and PixMo-Count. Together they cover free-form image questions, caption-derived QA, referring expressions and point annotations, explanations with inline points, synthetic documents and charts, clock reading, and counting.

</details>

### **VILA-U: Fully Autoregressive Visual Understanding and Generation**

VILA-U uses a shared visual tokenizer and a single next-token objective for image, video, and language understanding and generation, avoiding separate diffusion decoders or separate visual pathways.

[![arXiv](https://img.shields.io/badge/arXiv-2409.04429-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2409.04429) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mit-han-lab/vila-u)

VILA-U Team<br>
**Released:** 2024-09-06

<details>
<summary>ℹ️ <i>More Information</i></summary>

**VILA-U** converts images and video frames into discrete visual tokens and processes them together with text through one autoregressive language-model backbone. Its key component is a unified vision tower trained to preserve the information needed for both semantic understanding and visual reconstruction. Aligning its discrete visual tokens with language during pretraining improves perception while allowing the same tokens to be generated directly by next-token prediction.

Training mixes image-text understanding, video understanding, pure text, and high-quality text-to-image examples, then instruction-tunes the model for conversational and generative tasks. The design demonstrates that competitive image generation does not necessarily require a diffusion module and that understanding and generation can share a fully tokenized visual interface.

</details>

### **Show-o: Autoregressive Language and Discrete-Diffusion Vision in One Transformer**

Show-o unifies visual understanding, text generation, and image generation by combining causal autoregression with masked discrete diffusion inside a single transformer.

[![arXiv](https://img.shields.io/badge/arXiv-2408.12528-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.12528) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/showlab/Show-o) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/showlab/show-o)

Show Lab<br>
**Released:** 2024-08-22

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Show-o** represents images with discrete visual tokens and processes text and images through one omni-attention transformer. Language and multimodal-understanding sequences use autoregressive next-token prediction, whereas image generation uses iterative masked-token prediction, giving the same network a discrete-diffusion generation path. Its attention mask changes with the task: causal dependencies are retained where autoregression is required, while visual generation tokens can exchange bidirectional context.

Training combines large-scale text, image-text pairs, visual question answering and instruction data, and text-to-image data, followed by task-oriented fine-tuning. The resulting model supports captioning, VQA, text-to-image generation, inpainting, extrapolation, and interleaved mixed-modality generation without attaching an external diffusion model.

</details>

### **Transfusion: Next-Token Text Prediction and Continuous Image Diffusion**

Transfusion trains one transformer over mixed text-and-image sequences while applying the representation and loss best suited to each modality: autoregressive next-token prediction for text and diffusion for continuous image patches.

[![arXiv](https://img.shields.io/badge/arXiv-2408.11039-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.11039)

Meta FAIR<br>
**Released:** 2024-08-20

<details>
<summary>ℹ️ <i>More Information</i></summary>

Unlike unified models that quantize images into discrete codes, **Transfusion** retains continuous image representations and denoises them with a diffusion objective. Text tokens and noisy image patches occupy the same sequence; modality-specific input and output layers connect them to a shared transformer. Causal attention governs text, while image patches can interact bidirectionally within each image.

The paper derives scaling behavior from models pretrained from scratch and scales the recipe to 7B parameters and two trillion multimodal tokens. Additional modality-specific encoding and decoding layers improve image fidelity and can compress an image to as few as 16 latent patches. This makes Transfusion an important alternative to fully discrete, next-token-only multimodal architectures.

</details>

### **mPLUG-Owl3: Hyper-Attention for Long Image Sequences**

mPLUG-Owl3 keeps long visual inputs outside the ordinary language-token stream and lets language tokens retrieve relevant visual evidence through hyper-attention, making multi-image and long-video reasoning more efficient.

[![arXiv](https://img.shields.io/badge/arXiv-2408.04840-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.04840) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/X-PLUG/mPLUG-Owl) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat-square)](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)

mPLUG-Owl Team<br>
**Released:** 2024-08-09

<details>
<summary>ℹ️ <i>More Information</i></summary>

Instead of concatenating every vision feature with the language sequence and paying full self-attention cost, **mPLUG-Owl3** introduces hyper-attention blocks that use language states to selectively aggregate separately maintained visual features. These blocks are inserted into the language model and create a language-guided semantic space for single images, interleaved image-text documents, retrieved visual knowledge, and long videos.

The approach reduces interference from irrelevant frames and avoids consuming the entire language context with visual tokens. Training combines single-image instruction data with multi-image, interleaved, retrieval-augmented, and video examples. The accompanying Distractor Resistance evaluation measures whether the model can retain the relevant visual evidence when long sequences contain competing images.

</details>

### **Cambrian-1: Vision-Centric Multimodal LLMs**

Cambrian-1 is a family of 8B, 13B, and 34B multimodal LLMs centered on stronger visual representation and spatial grounding. Its Spatial Vision Aggregator dynamically combines complementary high-resolution features while limiting the number of visual tokens passed to the language model.

[![arXiv](https://img.shields.io/badge/arXiv-2406.16860-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.16860) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/cambrian-mllm/cambrian)

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, Saining Xie<br>
**Released:** 2024-06-24

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Cambrian-1** treats the language model as an interface for systematically studying visual representations. The paper evaluates more than twenty encoders and builds its strongest systems from complementary CLIP, SigLIP, DINOv2, and ConvNeXt features. Its **Spatial Vision Aggregator** uses learnable visual queries and auxiliary connectors to gather local, spatially aligned evidence from these feature grids. This preserves high-resolution information while producing a fixed, substantially smaller set of tokens for the LLM. Released variants pair the connector with Phi-3, Llama 3, Vicuna 1.5, or Hermes-2-Yi backbones.

Training has two stages. First, the vision encoders and LLM remain frozen while the Spatial Vision Aggregator is trained on approximately 2.5M alignment samples. Second, the connector and language model are instruction-tuned together on Cambrian-7M. The authors' scaling studies emphasize balanced data composition, limiting overrepresented sources, and retaining demanding vision-centric examples rather than simply maximizing sample count.

Cambrian-7M is the curated subset of a larger Cambrian-10M pool assembled from public visual-question-answering, visual-conversation, OCR, chart, counting, math, coding, science, and embodied-interaction sources. The project releases the models, code, alignment data, instruction data, training recipes, and CV-Bench, its vision-centric evaluation suite.

</details>

### **Ovis: Structural Visual-Text Embedding Alignment**

Ovis replaces the usual direct projection of continuous vision features with a learnable visual vocabulary, structurally aligning visual embeddings with the lookup-table mechanism used for text tokens.

[![arXiv](https://img.shields.io/badge/arXiv-2405.20797-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.20797) [![Ovis 2.5](https://img.shields.io/badge/arXiv-2508.11737-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.11737) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/AIDC-AI/Ovis) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Ovis2.5-blue?style=flat-square)](https://huggingface.co/AIDC-AI/Ovis2.5-9B)

Alibaba International Digital Commerce<br>
**Released:** 2024-06-14

<details>
<summary>ℹ️ <i>More Information</i></summary>

The original **Ovis** vision encoder predicts distributions over a learnable visual embedding table; weighted combinations of those embeddings become the visual tokens consumed by the language model. This mirrors textual embedding lookup more closely than a conventional MLP connector and makes the visual tokenizer and LLM independently replaceable. Ovis1.5 and 1.6 improved high-resolution processing, data quality, and preference alignment.

**Ovis2**, released January 26, 2025, paired an AIMv2-based visual tokenizer with Qwen2.5 backbones and expanded the family from 1B to 34B, adding stronger OCR, video, multi-image, and reasoning behavior. **Ovis2.5**, released August 15, 2025, moved to a native-variable-resolution SigLIP 2 vision transformer and Qwen3 backbones. Its five-stage curriculum progresses from visual and multimodal pretraining through instruction tuning, DPO, and GRPO, enabling an optional reflective thinking mode.

</details>

### **Phi-3-Vision and Phi-3.5-Vision: Compact Long-Context Multimodal Reasoning**

Phi-3-Vision brought chart, document, diagram, and image reasoning to a compact 4.2B-parameter model; Phi-3.5-Vision retained the small footprint while extending the family to multi-image inputs.

[![Technical Report](https://img.shields.io/badge/arXiv-2404.14219-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2404.14219) [![Official Release](https://img.shields.io/badge/Microsoft-Release-0078D4.svg?style=flat-square)](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Phi_3.5_Vision-blue?style=flat-square)](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

Microsoft Phi Team<br>
**Released:** 2024-05-21

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Phi-3-Vision** connects a vision encoder to the Phi-3 Mini language backbone through a learned projection and uses a dynamic high-resolution image pipeline, preserving detail through global and cropped image features while supporting a 128K-token context. Its 4.2B parameters make it substantially smaller than many contemporary VLMs.

Training combines carefully filtered public material, image-text and interleaved examples, synthetic reasoning data, and instruction data emphasizing OCR, charts, tables, diagrams, slides, and general visual reasoning. Microsoft released Phi-3-Vision on May 21, 2024. **Phi-3.5-Vision** followed on August 22, 2024, maintaining the same compact family design while improving visual quality, multilingual behavior, and multi-frame or multi-image reasoning.

</details>

### **Chameleon: Mixed-Modal Early-Fusion Foundation Models**

Chameleon is an early-fusion, token-based model that represents text and images in one discrete vocabulary and processes arbitrary interleaved sequences with a single autoregressive transformer. The same architecture can understand and generate either modality without a separate vision encoder attached to a language model.

[![arXiv](https://img.shields.io/badge/arXiv-2405.09818-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.09818) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/chameleon) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-blue?style=flat-square)](https://huggingface.co/facebook/chameleon-7b)

Chameleon Team<br>
**Released:** 2024-05-16

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Chameleon** converts images into discrete codes with a learned image tokenizer and combines them with text tokens in a unified vocabulary. A decoder-only transformer then models mixed sequences through next-token prediction, enabling text-to-text, image-to-text, text-to-image, and interleaved image-text generation within one architecture. Meta trains 7B- and 34B-parameter variants and introduces stabilization measures tailored to the mixed-modal setting, including query-key normalization, revised normalization placement, and modality-aware normalization.

Training proceeds through large-scale mixed-modal pretraining followed by an alignment stage. The pretraining run covers approximately 4.4 trillion tokens and mixes text-only sequences, image-text pairs, and interleaved documents. The alignment recipe uses supervised fine-tuning together with preference optimization to improve response quality and safety while preserving the model's ability to produce long mixed-modal sequences. The released artifacts provide inference support for the text-output models; the paper's full research system also evaluates image and mixed-modal generation.

The corpus combines licensed text, publicly available text, image-caption pairs, and interleaved web documents in which images occur within their textual context. Additional high-quality instruction and preference data are used during alignment. Keeping the modalities in their original interleaved order is essential: Chameleon is trained to model complete mixed-modal documents, rather than treating every image as an isolated attachment to a prompt.

</details>

### **MM1: Methods, Analysis, and Insights from Multimodal Pre-training**

MM1 is Apple's family of dense and mixture-of-experts multimodal language models, built from a pretrained image encoder, a vision-language connector, and a decoder-only language model. Its controlled ablations show that visual capacity and pretraining data composition matter substantially more than connector complexity.

[![arXiv](https://img.shields.io/badge/arXiv-2403.09611-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2403.09611) [![Project](https://img.shields.io/badge/Apple-Research-black?style=flat-square)](https://machinelearning.apple.com/research/mm1-methods-analysis-insights)

Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, Anton Belyi, Haotian Zhang, et al.<br>
**Released:** 2024-03-14

<details>
<summary>ℹ️ <i>More Information</i></summary>

**MM1** combines a pretrained vision encoder with a vision-language connector that maps image features into tokens consumed by a decoder-only language model. Apple studies dense models from 3B to 30B parameters alongside mixture-of-experts variants. Controlled experiments vary the image encoder, input resolution, number of visual tokens, connector, and language-model initialization. The results identify the image encoder, resolution, and visual-token budget as consequential choices, while the difference among connector designs is comparatively small once the other components are well configured.

Pretraining uses a deliberate mixture of image-caption pairs, interleaved image-text documents, and text-only sequences. The paper finds that caption data supports strong zero-shot behavior, interleaved documents are especially important for few-shot and in-context learning, and retaining text-only data protects language capability. The scaled recipe gives MM1 multi-image reasoning and few-shot chain-of-thought behavior before supervised fine-tuning; subsequent multimodal instruction tuning improves conversational and benchmark performance.

The pretraining corpus is organized into three corresponding data families: web-scale image-caption pairs, long interleaved documents containing multiple images and surrounding text, and large text-only corpora. The caption component includes both original and synthetically generated captions. Because the paper's principal contribution is a controlled study of mixture and architecture choices, it reports data by source type and experimental mixture rather than presenting a separately released named dataset.

</details>

### **AnyGPT: Unified Any-to-Any Multimodal Modeling with Discrete Tokens**

AnyGPT treats text, speech, images, and music as token sequences in one autoregressive language model, showing that new modalities can be incorporated largely through tokenization and data construction rather than architectural surgery.

[![arXiv](https://img.shields.io/badge/arXiv-2402.12226-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2402.12226) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenMOSS/AnyGPT)

OpenMOSS<br>
**Released:** 2024-02-19

<details>
<summary>ℹ️ <i>More Information</i></summary>

**AnyGPT** converts every supported modality into discrete tokens and trains a LLaMA-family decoder with the ordinary next-token objective. Modality-specific tokenizers encode and reconstruct speech, images, and music, while special tokens delimit each modality inside interleaved sequences.

Training begins with multimodal alignment data that connects modality tokens to text, then uses the authors' 108,000-example any-to-any instruction dataset for multi-turn conversations containing arbitrary input and output modalities. The central result is architectural simplicity: speech, image, music, and text generation share one vocabulary-level interface and one autoregressive backbone, without separate cross-attention modules or task-specific generation heads.

</details>

### **LLaVA: Large Language and Vision Assistant - Visual Instruction Tuning**

LLaVA seamlessly integrates a pre-trained language model (Vicuna) with a visual encoder (CLIP) using a simple linear layer, creating a robust architecture capable of effectively processing and understanding language-image instructions.

[![arXiv](https://img.shields.io/badge/arXiv-2304.08485-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.08485) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/haotian-liu/LLaVA)

Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/722f0fbb-ea52-4a8a-ab1e-bec45ca7d04f" alt="Architecture diagram for LLaVA: Large Language and Vision Assistant - Visual Instruction Tuning" />
</p>
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**LLaVA**: At the heart of LLaVA's architecture is the fusion of a pre-trained language model with a visual model, specifically designed to process and understand language-image instruction data effectively. This integration enables LLaVA to leverage the distinct strengths of both models, employing the CLIP visual encoder for robust image feature extraction and the Vicuna language model for intricate language instruction processing. A noteworthy feature of this architecture is the use of **a simple linear layer** that bridges image features to the word embedding space, facilitating a seamless alignment between visual and linguistic representations. The training methodology of LLaVA is meticulously structured into a two-stage instruction-tuning procedure. Initially, the model undergoes pre-training focused on feature alignment, utilizing a carefully filtered dataset to synchronize image features with LLM word embeddings. Subsequently, the model is fine-tuned end-to-end on tailored tasks such as multimodal chatbot functionalities and Science QA, with the aim of refining its instruction-following prowess. This sophisticated training regimen is underpinned by the use of multimodal instruction-following data generated via GPT-4, converting image-text pairs into formats conducive to instruction-following tasks. The alignment of text and image data is innovatively achieved through **a trainable projection matrix**, converting visual features into language embedding tokens within a unified dimensional space, thereby enhancing the model's ability to encode vision and text cohesively.The datasets deployed for LLaVA's training and evaluation are strategically selected to bolster its multimodal capabilities. The Filtered CC3M dataset serves as the foundation for pre-training, aligning visual and language features, while the LLaVA-Instruct-158K dataset generated using GPT-4 is pivotal for fine-tuning the model on diverse multimodal tasks. Additionally, the ScienceQA dataset plays a critical role in assessing LLaVA's proficiency in multimodal reasoning tasks, demonstrating the model's comprehensive training and its potential to significantly advance the field of multimodal interaction and understanding.
</details> 

### **LLaVA 1.5: Improved Baselines with Visual Instruction Tuning**

LLaVA 1.5 enhances its multimodal understanding by replacing its initial linear projection with a more powerful multi-layer perceptron (MLP), enabling a deeper integration of visual features from CLIP-ViT-L-336px and linguistic data.

[![arXiv](https://img.shields.io/badge/arXiv-2310.03744-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.03744)  
Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c7112b75-3b86-48a2-9c0f-f1dc1dc6ee06" alt="Architecture diagram for LLaVA 1.5: Improved Baselines with Visual Instruction Tuning" />
</p>
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA 1.5**: This iteration introduces a refined architecture that incorporates a CLIP-ViT-L-336px vision encoder alongside **a multi-layer perceptron (MLP) projection layer**. This combination not only boosts the model's data efficiency but also its performance across various benchmarks, showcasing a leap in multimodal understanding. The architecture's core components, the CLIP-ViT-L for visual encoding and the MLP for vision-language cross-modal connection, work synergistically to enhance the model's capacity to integrate and interpret visual and linguistic inputs.Training methods have been optimized in LLaVA 1.5 to achieve unprecedented performance on 11 benchmarks, utilizing a two-stage approach that emphasizes efficient feature alignment and fine-tuning with VQA data specifically tailored for academic tasks. The paper highlights a shift towards more sophisticated multimodal alignment techniques, **replacing the original linear projection** with a more powerful **MLP vision-language connector**. This strategic improvement facilitates a deeper and more nuanced integration of visual and linguistic data. Moreover, the adoption of an MLP-based vision-language connector for alignment fusion methods further strengthens the model's ability to merge visual and textual representations effectively, ensuring closer alignment in the embedding space.The utilization of datasets such as VQA-v2, GQA, and other academic-task-oriented VQA datasets, enriched with OCR and region-level perception data, underscores the model's enhanced visual understanding and reasoning capabilities. These datasets play a crucial role in elevating LLaVA 1.5's performance, enabling it to set new standards with academic-task-oriented data. Through these advancements, LLaVA 1.5 not only pushes the boundaries of multimodal learning but also sets a new benchmark for future research in the field.
</details> 

### **LLaVA 1.6: LLaVA-NeXT Improved reasoning, OCR, and world knowledge**

LLaVA-NeXT advances on LLaVA-1.5 by incorporating high-resolution image processing, enhancing visual reasoning and OCR capabilities, while maintaining a data-efficient design through knowledge transfer from its predecessor and a refined training process.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://llava-vl.github.io/blog/2024-01-30-llava-next/)  
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/032ef144-ec10-41da-80a1-2cecd66c86ee" alt="Architecture diagram for LLaVA 1.6: LLaVA-NeXT Improved reasoning, OCR, and world knowledge" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA-NeXT**: Represents a significant step forward in the evolution of large language models with visual capabilities, building upon the foundations laid by LLaVA-1.5. This model introduces several enhancements aimed at improving image resolution, visual reasoning, optical character recognition (OCR), and the integration of world knowledge, all while retaining the minimalist and data-efficient design of its predecessor. The architecture of LLaVA-NeXT is optimized for high performance, supporting input image resolutions up to 672x672, 336x1344, and 1344x336 pixels. This improvement facilitates a more detailed visual perception, which, coupled with an enhanced visual instruction tuning data mixture, significantly bolsters the model's reasoning and OCR capabilities. Furthermore, LLaVA-NeXT achieves efficient deployment through the use of SGLang, a feature that underscores its design's focus on performance and data efficiency.Training LLaVA-NeXT requires less than 1 million visual instruction tuning samples, leveraging the **pre-trained connector** from LLaVA-1.5 for efficient knowledge transfer. The training process, remarkably swift, utilizes 32 A100 GPUs and completes in approximately one day, a testament to the model's efficient design and deployment strategy. The alignment techniques in LLaVA-NeXT are particularly noteworthy, utilizing high-resolution images and a high-quality data mixture to enhance the model's capabilities in visual conversation and instruction following. The model's use of dynamic high-resolution techniques, known as 'AnyRes', allows for effective handling of images with varying resolutions, improving the model's overall visual understanding.The datasets employed in training LLaVA-NeXT, including LAION-GPT-V, ShareGPT-4V, DocVQA, SynDog-EN, ChartQA, DVQA, and AI2D, are meticulously chosen to augment the model's visual reasoning, OCR capabilities, and comprehension of charts and diagrams. This strategic selection aims to elevate the model's performance across a wide range of multimodal tasks, emphasizing its enhanced ability to process and understand complex visual information. Through these improvements, LLaVA-NeXT sets a new benchmark for models at the intersection of language and vision, offering unprecedented capabilities in visual reasoning, OCR, and the application of world knowledge in multimodal contexts.
</details> 

### **PaliGemma: A Versatile and Transferable 3B Vision-Language Model**

PaliGemma is a compact, open-source vision-language model designed to be easily transferable to a diverse range of tasks. It combines a powerful SigLIP image encoder with the Gemma-2B language model, achieving strong performance on over 40 diverse tasks, including standard VLM benchmarks, remote-sensing, and segmentation. PaliGemma is pretrained using a multi-stage approach, focusing on maximizing the density of learning signal and providing different checkpoints with varying image resolutions. This versatile foundation model is easily fine-tuned for specific tasks and serves as a valuable tool for researchers and practitioners exploring the capabilities of VLMs.

[![arXiv](https://img.shields.io/badge/arXiv-2407.07726-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2407.07726) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/big-vision/paligemma)  
Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, Xiaohua Zhai  

<p align="center">
<img src="https://github.com/user-attachments/assets/186371d0-6861-4b68-b32e-fee77cc75ef2" alt="Architecture diagram for PaliGemma: A Versatile and Transferable 3B Vision-Language Model" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

PaliGemma stands out as a highly versatile and transferable 3-billion parameter Vision-Language Model (VLM) meticulously designed for broad applicability across a wide spectrum of visual-language tasks. Its foundation lies in the integration of two powerful components: a SigLIP-So400m vision encoder, known for its exceptional performance despite its compact size, and the Gemma-2B language model, a pretrained autoregressive decoder-only model from the Gemma family. This combination enables PaliGemma to effectively process and understand both visual and textual information, making it adept at handling tasks ranging from image captioning and visual question answering to more specialized tasks like remote-sensing and segmentation. PaliGemma's architecture is streamlined and efficient. It uses a simple linear projection to align the visual features extracted by the SigLIP encoder with the vocabulary tokens of the Gemma language model, enabling seamless fusion of the two modalities. A key aspect of PaliGemma's training is the emphasis on "density of learning signal," prioritizing a broad range of skills and knowledge over achieving high zero-shot performance. This approach involves a multi-stage pretraining process that starts with unimodal pretraining of individual components using publicly available checkpoints, followed by extensive multimodal pretraining on a diverse mixture of large-scale vision-language tasks. Notably, PaliGemma deviates from the common practice of freezing the image encoder during pretraining, allowing it to learn spatial and relational understanding from complex tasks like captioning. To further enhance its capabilities, PaliGemma undergoes a resolution increase stage, where it is trained on higher-resolution images, enabling it to handle tasks that benefit from finer visual details. This multi-stage pretraining process results in a family of three PaliGemma checkpoints at varying image resolutions (224px, 448px, and 896px), each pretrained with broad visual knowledge. These checkpoints serve as strong base models that can be easily transferred to specific downstream tasks. PaliGemma's transferability is demonstrated through its impressive performance on over 30 academic benchmarks, including those involving multiple images, such as NLVR2 and short-video understanding tasks. The model's ability to adapt quickly to new tasks with minimal fine-tuning highlights its versatility and makes it a valuable tool for exploring and advancing the capabilities of VLMs. Furthermore, the model's open-source nature, along with its straightforward architecture and training recipe, encourages further research and experimentation within the VLM community, driving progress towards more powerful and general-purpose multimodal AI systems.
</details>

### **PaliGemma 2: A Family of Versatile VLMs for Transfer**

PaliGemma 2 is an upgraded family of open Vision-Language Models (VLMs) based on Gemma 2 language models, combined with the SigLIP-So400m vision encoder. It offers models in three sizes (3B, 10B, 28B) and three resolutions (224px², 448px², 896px²), trained in multiple stages for broad knowledge transfer.  PaliGemma 2 achieves state-of-the-art results on various tasks, including OCR-related challenges like table/molecular/music score recognition, and long-form captioning.

[![arXiv](https://img.shields.io/badge/arXiv-2412.03555-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.03555)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)  
Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer and Xiaohua Zhai

<p align="center">
<img src="https://github.com/user-attachments/assets/4ce402be-d644-4143-a57c-9e7f4d811d95" width="600" alt="Architecture diagram for PaliGemma 2: A Family of Versatile VLMs for Transfer" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

PaliGemma 2 closely follows the architecture of its predecessor, PaliGemma.  It uses a pre-trained SigLIP-So400m vision encoder. The embeddings from this encoder are mapped to the input space of the Gemma 2 language model using a *linear projection*.  The combined visual and text embeddings are then fed into the Gemma 2 model, which autoregressively generates the output.  The model comes in three size variants (2B, 9B, and 27B parameters in the Gemma 2 component, corresponding to 3B, 10B, and 28B total parameters) and is trained at three resolutions (224x224, 448x448, and 896x896 pixels).  This allows for analysis of the interplay between model size, resolution, and transfer performance. The input image gets concatenated with the input text tokes and Gemma 2 autoregressively completes this prefix with an answer. PaliGemma 2's training follows a three-stage approach, similar to the original PaliGemma: **Stage 1:** The pre-trained SigLIP-So400m and Gemma 2 checkpoints are combined and trained jointly on a multimodal task mixture of 1 billion examples. The image resolution is 224px². **Stage 2:** Training continues for 50 million examples at 448px² resolution, then for 10 million examples at 896px². Tasks benefiting from higher resolution are upweighted. **Stage 3:** Fine-tuning the checkpoints from stage 1 or 2 on the target tasks. The training data mixture includes captioning, grounded captioning, OCR, visual question answering (VQA), detection, and instance segmentation. Notably, the training data relies heavily on *machine-generated labels* from publicly available specialist models, *avoiding the use of large commercial VLMs* for label generation. **Gemma 2 Language Models:**  The core upgrade is the use of the more recent and capable Gemma 2 family of language models, replacing the original Gemma model in PaliGemma.  **Resolution and Model Size Scaling:** PaliGemma 2 systematically explores the impact of both image resolution and language model size on transfer performance.  This is a key contribution, as most prior work did not jointly study these factors with consistent training recipes.
</details>

### **AIMv2: Multimodal Autoregressive Pre-training of Large Vision Encoders**

AIMv2 is a family of generalist vision encoders that autoregressively generates both image patches and text tokens, achieving state-of-the-art performance in multimodal image understanding and strong results in vision benchmarks like localization, grounding, and classification, demonstrating scalability and efficiency.

[![arXiv](https://img.shields.io/badge/arXiv-2411.14402-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2411.14402)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/apple/ml-aim)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/apple/aimv2-large-patch14-224)  
Enrico Fini, Mustafa Shukor, David Haldimann, Sai Aitharaju, Alexander Toshev, Marcin Eichner, Moin Nabi, Xiujun Li, Philipp Dufter, Michal Klein, Victor G. Turrisi da Costa, Louis Béthune, Zhe Gan, Alaaeldin El-Nouby

<p align="center">
<img src="https://github.com/user-attachments/assets/c89a0be9-8743-4800-8d3c-ec51a4c99f4d" width="600" alt="AIMv2 architecture" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>


AIMv2 (Autoregressive Image Models v2) introduces a novel pre-training method for large-scale vision encoders that extends autoregressive pre-training to a multimodal setting, encompassing both images and text. The core architecture pairs a Vision Transformer (ViT) encoder with a causal multimodal decoder. The vision encoder processes raw image patches (using prefix attention), while the multimodal decoder autoregressively generates both image patches (using pixel MSE loss) and text tokens (using cross-entropy loss). Crucially, image patches and text tokens are treated as a single, unified sequence. This allows the model to learn a joint representation of visual and textual information. The image is always prepended to the beginning of text sequence. The training process is streamlined and efficient. It resembles that of AIM and LLMs, relying solely on the autoregressive objective. There are no specialized inter-batch communication methods or excessively large batch sizes are required. This contrasts with contrastive methods (e.g., CLIP, SigLIP), which are often more challenging to train and scale. The training data consists of a mixture of publicly available (DFN-2B, COYO) and proprietary datasets (HQITP), comprising both alt-text and synthetic captions. AIMv2 demonstrates strong scaling properties, consistently improving performance with increased data or model parameters. The model family includes variants ranging from 300 million to 3 billion parameters. A key optimization is the use of prefix attention within the vision encoder, enabling bidirectional attention during inference without fine-tuning. Other architectural choices include the incorporation of SwiGLU and RMSNorm, inspired by recent successes in language modeling. AIMv2 excels in a variety of tasks. It performs favorably on multimodal understanding benchmarks compared to state-of-the-art vision-language pre-trained methods . It also exhibits strong performance on open-vocabulary object detection and referring expression comprehension, surpassing DINOv2. Additionally, it achieves impressive recognition performance with a frozen trunk. The model supports native image resolution and adaptation to zero-shot recognition, demonstrating its flexibility. Post-training strategies, including high-resolution adaptation, further enhance the model's capabilities. Ablation studies demonstrate the importance of joint image and text modeling, validate design choices, and explore scaling characteristics.
</details>

### **Apollo: An Exploration of Video Understanding in Large Multimodal Models**


Apollo is a family of Large Multimodal Models (LMMs) designed for video understanding. Its study introduces "Scaling Consistency" and examines video sampling, architecture, data composition, and training schedules; at publication, the authors reported that Apollo-3B outperformed most compared 7B models.

[![arXiv](https://img.shields.io/badge/arXiv-2412.10360-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.10360)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://apollo-lmms.github.io/)  
Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta, Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang Wang, Felix Juefei-Xu, Ning Zhang, Serena Yeung-Levy, Xide Xia

<p align="center">
<img src="https://github.com/user-attachments/assets/9222064a-d7a3-4e6b-a14d-bc9a5c679450" width="600" alt="Architecture diagram for Apollo: An Exploration of Video Understanding in Large Multimodal Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>


Apollo leverages the Qwen2.5 series of Large Language Models (LLMs) with 1.5B, 3B, and 7B parameters. The key architectural innovation is the combination of a SigLIP-SO400M image encoder and an InternVideo2 video encoder. Features from both encoders are interpolated and concatenated channel-wise before being fed into a Perceiver Resampler, which outputs 32 tokens per frame. This combination was empirically found to be superior to other encoder choices. The model uses a 3-stage training approach. Critically, the paper introduces the concept of "Scaling Consistency," demonstrating that design decisions made on smaller models and datasets (up to a critical size) effectively transfer to larger models. This allows for more efficient experimentation. The paper also advocates for frames-per-second (fps) sampling during training, as opposed to uniform frame sampling, and demonstrates its superiority. The optimal number of tokens is 8-32 per frame. It also includes a curated benchmark, ApolloBench, that reduces evaluation time by 41x compared to existing benchmarks while maintaining high correlation and focusing on temporal reasoning and perception. The exploration also includes Token Resampling showing that Perceiver resampling has a good performace. Token Integration is also discussed: Adding tokens (text, learned, etc.) between the video tokens derived from different frames
or clips is sufficient for efficient token integration. Training Stages is also disscussed, concluding that progressively unfreezing the different components in different stages leads to superior model training dynamics. Finally, training the Video Encoder is discussed. The paper concludes that Finetuning video encoders on only video data further improves overall performance,
especially on reasoning and domain-specific tasks. Data Composition is also studied. It concludes that Data mixture matters, and including a moderate amount of text data and maintaining a
slight video-heavy mix leads to optimal performance.

</details>

### **ARIA: An Open Multimodal Native Mixture-of-Experts Model**

ARIA is an open-source, multimodal native Mixture-of-Experts (MoE) model designed to seamlessly integrate and understand diverse modalities like text, code, images, and video, achieving state-of-the-art performance in its class. It features a fine-grained MoE decoder for efficient parameter utilization, a lightweight visual encoder, and a 4-stage training pipeline that builds capabilities in language understanding, multimodal comprehension, long context handling, and instruction following.

[![arXiv](https://img.shields.io/badge/arXiv-2410.05993-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2410.05993)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/rhymes-ai/Aria)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/blog/RhymesAI/aria)  
Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, Chongyan Zhu, Xiaoyi Ren, Chao Li, Yifan Ye, Peng Liu, Lihuan Zhang, Hanshu Yan, Guoyin Wang, Bei Chen, Junnan Li

<p align="center">
<img src="https://github.com/user-attachments/assets/efe4a7ba-756a-4da8-b261-5a0093f28b03" width="600" alt="Architecture diagram for ARIA: An Open Multimodal Native Mixture-of-Experts Model" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

ARIA's architecture is centered around a fine-grained Mixture-of-Experts (MoE) decoder, which is more efficient than traditional dense decoders.  This MoE approach activates 3.5B parameters per text token and 3.9B per visual token, out of a total of 24.9B parameters. The model uses 66 experts in each MoE layer, with 2 shared across all inputs for common knowledge, and 6 activated per token by a router. The visual encoder is a lightweight (438M parameter) Vision Transformer (ViT) combined with a projection module.  The ViT processes images at various resolutions (medium, high, and ultra-high), preserving aspect ratios. The projection module uses cross-attention and an FFN layer to convert image embeddings into visual tokens, which are then integrated with text tokens by the MoE. ARIA's training uses a 4-stage pipeline: (1) Language pre-training (6.4T text tokens, 8K context window); (2) Multimodal pre-training (400B multimodal tokens, including interleaved image-text, synthetic image captions, document transcriptions and QA, video captions and QA); (3) Multimodal long-context pre-training (extending context to 64K tokens); and (4) Multimodal post-training (instruction following with 20B tokens).  The data curation process is rigorous, incorporating techniques like de-duplication, quality filtering, and data clustering. The training infrastructure avoids pipeline parallelism, using a combination of expert parallelism and ZeRO-1 data parallelism, which contributes to efficient training without the need for tensor parallelism. A load-balancing loss and z-loss are used to stabilize training.
The paper demonstrates that, despite having modality-generic experts, ARIA naturally develops expert specialization during pre-training.  Analysis of expert activation shows distinct visual specialization in several layers, particularly for image, video, and PDF content.  ARIA also shows excellent performance in handling long-context multimodal data, surpassing other open models and competing favorably with proprietary models in tasks like long video and document understanding.
</details>



### **EVE: Unveiling Encoder-Free Vision-Language Models**

EVE is an encoder-free vision-language model (VLM) that directly processes images and text within a unified decoder-only architecture, eliminating the need for a separate vision encoder. It achieves competitive performance with encoder-based VLMs of similar size on multiple vision-language benchmarks using only 35M publicly accessible data, with the model efficiently handling high-resolution images with arbitrary aspect ratios.

[![arXiv](https://img.shields.io/badge/arXiv-2406.11832-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.11832)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/baaivision/EVE)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/BAAI/EVE-7B-HD-v1.0)  
Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, Xinlong Wang
<p align="center">
<img src="https://github.com/user-attachments/assets/c10e987d-9e11-41d7-968c-617b60d3b0bd" width="600" alt="Architecture diagram for EVE: Unveiling Encoder-Free Vision-Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**EVE (Encoder-free Vision-language modEl)**:  This model distinguishes itself by completely removing the vision encoder component typically found in VLMs.  Instead, it directly integrates visual information into a decoder-only architecture (based on Vicuna-7B).  This is achieved through a novel **Patch Embedding Layer (PEL)** that processes image patches directly, combined with a **Patch Aligning Layer (PAL)** that facilitates learning from a pre-trained vision encoder (CLIP-ViT-L/14) without updating the encoder itself.  Crucially, EVE does *not* use a traditional image encoder during inference. The **PEL** uses a convolution layer and average pooling to create 2D feature maps from the input image.  It then employs cross-attention (CA1) within a limited receptive field to enhance these features. A special `<CLS>` token provides a holistic view of each patch feature, and a learnable newline token `<SPL>` is inserted after each row of patch features to represent the 2D structure. The **PAL** aligns EVE's patch features with those from a frozen, pre-trained vision encoder (CLIP-ViT-L/14). This is done hierarchically, aggregating features across multiple layers of the decoder and using a layer-wise cross-attention (CA3) mechanism.  A Mean Squared Error (MSE) loss between EVE's features and the vision encoder's features encourages alignment.  This "implicit" supervision from the vision encoder improves visual understanding.  Importantly, PAL is *only* used during training, not inference. The training process occurs in three stages: **LLM-guided Pre-training:**  Only the PEL and PAL are trained, aligning the visual features with the frozen LLM (Vicuna-7B).  This stage uses a subset (16M) of the total training data. **Generative Pre-training:** The entire model (including the LLM) is trained, using the full 33M dataset. Both text prediction (cross-entropy loss) and visual alignment (MSE loss) are used. **Supervised Fine-tuning:**  The entire model is fine-tuned on instruction-following datasets (LLaVA-mix-665K and others). The key innovations that allow EVE to work well without a vision encoder are: **LLM-Centric Pre-alignment:**  Stage 1 is critical for preventing model collapse and accelerating convergence.  Aligning visual features *before* fully training the LLM is essential. **Vision Recognition Capability via Extra Supervision:** The PAL provides supervision from a pre-trained vision encoder during training, which enhances visual understanding without requiring the encoder during inference. **Flexible Input Handling:**  The architecture naturally handles images of arbitrary aspect ratios and resolutions, without needing resizing, padding, or partitioning. No reliance on vision encoder: The image are directly input into the LLM model.
EVE uses a curated dataset of 33M publicly available image-text pairs for pre-training, with captions generated by Emu2 and LLaVA-1.5. Supervised fine-tuning utilizes datasets like LLaVA-mix-665K, AI2D, DocVQA, and others.
</details>

### **EVEv2: Improved Baselines for Encoder-Free Vision-Language Models**

EVEv2 represents a significant advancement in encoder-free vision-language models (VLMs), addressing limitations of previous approaches by introducing a "Divide-and-Conquer" architecture that maximizes scaling efficiency, reduces inter-modality interference, and achieves strong performance with superior data efficiency.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/BAAI/EVE-7B-HD-v2.0)  
Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, Xinlong Wang

<p align="center">
<img src="https://github.com/user-attachments/assets/23a33fe6-d4c5-4a9d-b45f-f5612f7848a5" width="600" alt="Architecture diagram for EVEv2: Improved Baselines for Encoder-Free Vision-Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

EVEv2 departs from the traditional encoder-based VLM approach.  Instead of relying on a pre-trained vision encoder (like CLIP), it builds visual perception *directly within* a decoder-only Large Language Model (LLM).  Key architectural features include: **Divide-and-Conquer:** This is the core innovation.  Instead of mixing visual and textual information throughout the entire LLM, EVEv2 introduces *modality-specific* components.  This means separate attention matrices (query, key, value), Layer Normalization layers, and Feed-Forward Networks for visual and textual tokens.  This reduces interference and allows for more efficient learning.  It's a fully sparse, decoder-only architecture. **Patch Embedding Layer:**  A minimalist patch embedding layer is learned *from scratch*.  This avoids the inductive biases of pre-trained vision encoders.  It uses two convolutional layers (Conv1 and Conv2) to process image patches. **Lossless Encoding:**  Unlike some encoder-free models that use discrete tokenization (which can lose information), EVEv2 aims for lossless encoding of visual information. **LLM Adaptation:** The architecture is designed for seamless adaptation to existing LLMs.  The paper experiments with Vicuna-7B and Qwen2-7B. **Multi-Stage Training:** A four-stage training process is used: **LLM-guided Pre-aligning:**  Only the patch embedding layer is trained, using re-captioned web data (EVE-recap-10M).  The LLM is frozen. This establishes a basic alignment between visual and textual representations. **Vision Perception Learning:** Vision layers within the LLM are trained, using progressively larger datasets and image resolutions.  The LLM weights are still frozen. **Vision-Text Fully alligning:** The entire network is update. **Supervised Fine-tuning (SFT):**  The entire model is fine-tuned on question-answering and instruction-following datasets. **DenseFusion++:** A new, efficient captioning engine is introduced to generate high-quality image-text pairs for training.  This is crucial for building strong visual perception from scratch. It leverages multiple vision experts. **Data Efficiency:** A key focus of the research is demonstrating that EVEv2 can achieve strong performance with *less* data than comparable encoder-based models, thanks to its efficient architecture.
</details>




### **Janus and Janus-Pro: Decoupled Visual Understanding and Generation**

Janus decouples the visual pathways used for understanding and generation while sharing one autoregressive transformer; Janus-Pro scales and improves that architecture through a revised curriculum and substantially larger data mixtures.

[![Janus](https://img.shields.io/badge/arXiv-2410.13848-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2410.13848) [![Janus Pro](https://img.shields.io/badge/arXiv-2501.17811-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.17811)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/deepseek-ai/Janus)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/deepseek-ai/Janus-Pro-7B)  
DeepSeek-AI<br>
**Released:** 2024-10-17

<p align="center">
<img src="https://github.com/user-attachments/assets/657b0f2a-7a0e-4aed-a214-a33485990790" width="600" alt="Architecture diagram for Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Janus** uses a unified autoregressive transformer but separate visual encoders for understanding and generation. A SigLIP encoder and understanding adaptor supply semantic features, while a VQ tokenizer and generation adaptor turn images into discrete codes. This separation prevents the fine-grained reconstruction demands of image generation from compromising the high-level representations needed for comprehension.

**Janus-Pro**, released in January 2025, retains that split and improves it through a three-stage curriculum, more balanced multimodal, text, and generation mixtures, and scaling from 1.5B to 7B language backbones. Its expanded data includes roughly 90M understanding examples and 72M synthetic aesthetic generation examples. Treating both releases as one lineage preserves the original architectural contribution rather than presenting the later checkpoint as the family's beginning.
</details>

### **LLaVA-CoT: Let Vision Language Models Reason Step-by-Step**

LLaVA-CoT is a novel Vision-Language Model (VLM) designed to perform autonomous, multi-stage reasoning, enabling it to tackle complex visual question-answering tasks by independently engaging in sequential stages of summarization, visual interpretation, logical reasoning, and conclusion generation.

[![arXiv](https://img.shields.io/badge/arXiv-2411.10440-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2411.10440)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/PKU-YuanGroup/LLaVA-CoT)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/Xkev/Llama-3.2V-11B-cot)  
Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, Li Yuan

<p align="center">
<img src="https://github.com/user-attachments/assets/1a5e32f0-4ffc-4514-8401-25777c2fac10" width="600" alt="Architecture diagram for LLaVA-CoT: Let Vision Language Models Reason Step-by-Step" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

LLaVA-CoT builds upon the Llama-3.2-Vision model and introduces a structured, four-stage reasoning process: Summary (briefly outlines the task), Caption (describes relevant image parts), Reasoning (detailed analysis), and Conclusion (provides the final answer).  Each stage is marked with specific tags (<SUMMARY>, <CAPTION>, <REASONING>, <CONCLUSION>) to maintain clarity. Unlike traditional Chain-of-Thought (CoT) prompting, LLaVA-CoT promotes structured thinking by first organizing the problem and known information, then performing detailed reasoning, and finally deriving a conclusion. The model is trained on the newly compiled LLaVA-CoT-100k dataset. This dataset integrates samples from various visual question answering sources and providing structured reasoning instructions. The dataset contains 99k image and Question answer pairs using GPT-4o to provide details. Data is gathered from general VQA datasets (ShareGPT4V, ChartQA, A-OKVQA, DocVQA, PISC, CLEVR) and Science targeted VQA (AI2D, GeoQA+, ScienceQA, CLEVR-Math).  The paper also proposes a novel inference-time stage-level beam search method. This method generates multiple candidate results at *each* stage of the reasoning process, selecting the best to continue, improving performance and scalability. This contrasts with traditional best-of-N or sentence-level beam search. The entire model is trained using the Supervised-Fine Tuning.
</details>

### **LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation**

LLM2CLIP is a fine-tuning approach which integrates Large Language Models (LLMs) with pre-trained CLIP visual encoders. It improves the model by using the LLM's ability to proccess and understant long captions, open-world knowledge.

[![arXiv](https://img.shields.io/badge/arXiv-2411.04997-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2411.04997)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/LLM2CLIP)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/microsoft/LLM2CLIP-EVA02-B-16)  
Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Xiyang Dai, Dongdong Chen, Chong Luo, Lili Qiu

<p align="center">
<img src="https://github.com/user-attachments/assets/44d6952e-98ea-4875-bd9c-0a09a683bcbb" width="600" alt="Architecture diagram for LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

LLM2CLIP is fine-tuning approach. It integrates LLM (Large Language Models) to already pretrained CLIP visual encoders. The main problem which is tried to be solved is that; LLM's text understanding capability is not reflected on CLIP models. The authors highlight that directly incorporating LLMs into CLIP often fails due to the poor separability of LLM output features.  To tackle this, they introduce a two-stage approach. **Stage 1: Caption Contrastive (CC) Fine-tuning:** The LLM (specifically Llama-3 8B) is fine-tuned using a contrastive learning framework on a dataset of image captions (CC3M). This stage *doesn't train for autoregressive capabilities*, instead, it is transforming the causal attention to bidirectional, to function it as an encoder. This stage aims to improve the discriminative power of the LLM's output space, making it easier to distinguish between different captions, using supervised SimCSE loss. **Stage 2: CLIP Vision Encoder Fine-tuning:**  The pre-trained CLIP visual encoder is fine-tuned using the CC-fine-tuned LLM, now acting as a "super" text encoder.  The LLM's gradients are *frozen* during this stage to preserve its acquired knowledge and reduce computational cost.  Learnable adapters (linear layers) are added after the LLM to facilitate alignment with the CLIP visual encoder. 
Instead of the typical image-text contrastive loss, a caption-to-caption contrastive framework is used during LLM fine-tuning.  This forces the LLM to produce distinct representations for different captions describing the same image. It uses Supervised SimCSE. Makes the model encoder. Freezing the LLM during CLIP fine-tuning is crucial for efficiency and preserving the LLM's knowledge. These adapters bridge the gap between the frozen LLM and the CLIP visual encoder. The method is surprisingly efficient, requiring only a small amount of open-source data (15M or even 3M image-text pairs) and a single epoch of training in some cases. It leverages LoRA (Low-Rank Adaptation) for efficient fine-tuning. LLM2CLIP can effectively leverage dense captions (detailed image descriptions), a known limitation of standard CLIP. Uses ShareCaptioner-modified CC-3M (for CC fine-tuning), Wikitext-103, and a combination of CC-3M, CC-12M, YFCC-15M, and Recaption-1B for CLIP fine-tuning. The paper demonstrates that, after fine-tuning of the output space of the LLM, using LLM has a significant impact and it substantially improves the performance on downstream tasks.
</details>

### **Maya: An Instruction Finetuned Multilingual Multimodal Model**

Maya is an open-source Multimodal Multilingual Vision Language Model (mVLM) designed to address the limitations of current VLMs in handling low-resource languages and diverse cultural contexts, achieved by creating a new multilingual image-text pretraining dataset, performing toxicity analysis and mitigation, and fine-tuning for enhanced cultural and linguistic comprehension.

[![arXiv](https://img.shields.io/badge/arXiv-2412.07112-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.07112)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/nahidalam/maya)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/maya-multimodal/maya)  
Nahid Alam, Karthik Reddy Kanjula, Bala Krishna S Vegesna, S M Iftekhar Uddin, Drishti Sharma, Abhipsha Das, Shayekh Bin Islam, Surya Guthikonda, Timothy Chung, Anthony Susevski, Ryan Sze-Yin Chan, Roshan Santhosh, Snegha A, Chen Liu, Isha Chaturvedi, Ashvanth.S, Snehanshu Mukherjee, Alham Fikri Aji

<p align="center">
<img src="https://github.com/user-attachments/assets/f413afd9-3eee-4a5e-940a-b148fdf3189b" width="600" alt="Architecture diagram for Maya: An Instruction Finetuned Multilingual Multimodal Model" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Architecture:** Maya builds upon the LLaVA 1.5 framework.  It uses the Aya-23 8B model as its Large Language Model (LLM) due to Aya's strong multilingual capabilities (trained on 23 languages).  Critically, it *replaces* the CLIP vision encoder used in LLaVA with SigLIP.  This is motivated by SigLIP's superior performance, multilingual support, and ability to handle variable-length image patches (allowing for more flexible input sizes).  The visual features from SigLIP (`Zv = g(Xv)`) are passed through a trainable projection matrix (`W`, a 2-layer MLP with GELU activation) to align them with the LLM's embedding space, producing visual features `Hv`.  The architecture is fairly standard for this type of model, concatenating visual and textual features for input to the LLM. The training process involves two main phases: pretraining and finetuning. **Pretraining:**  The model is pretrained on a newly created multilingual image-text dataset.  This dataset is derived from the English-only LLaVA pretraining dataset (558k image-text pairs) and translated into seven additional languages (Chinese, French, Spanish, Russian, Hindi, Japanese, and Arabic) using a sophisticated translation pipeline.  This pipeline uses the Aya 35B model, optimized prompt engineering (determined empirically using BLEU and N-gram scores), and a batch processing approach with quality checks. Crucially, this dataset undergoes *toxicity filtering*.  LLaVAGuard and Toxic-BERT are used to identify and remove toxic image-caption pairs, creating a "toxicity-free" version of the dataset (removing 7,531 toxic images). The pretraining uses a learning rate of 1e-3 and a cosine scheduler. Only the projection matrix is trained during pretraining. **Finetuning:**  The pretrained model is then instruction-tuned using the PALO 150K instruction-tuning dataset (which covers 10 languages).  Full finetuning is performed (as opposed to LoRA), with frozen vision encoder and LLM. The core alignment technique is the trainable projection matrix (the 2-layer MLP) that maps the SigLIP visual features into the embedding space of the Aya-23 LLM. This is a simple but effective method, common in many VLMs. The paper *explicitly* states they did *not* use more complex alignment techniques like gated soft-attention (Flamingo) or Q-Former (BLIP-2) in this phase, reserving those for future work. **Pretraining Dataset:** A new multilingual dataset created by translating and filtering the LLaVA pretraining dataset.  This dataset is a key contribution of the paper.  The translation process and toxicity filtering are described in detail. **Instruction Tuning Dataset:** PALO 150K instruction-tuning dataset. **Evaluation Datasets**: PALO multilingual evalution, VizWiz, GQA, ScienceQA, TextVQA, POPE, MMBench, MM-Vet, MME. **Multilingual Image-Text Pretraining Dataset:** A new dataset of 558,000 images in eight languages. **Toxicity Analysis and Mitigation:**  A thorough analysis of toxicity in the original LLaVA dataset and the creation of a toxicity-free version. This is a significant and novel aspect. **Multilingual Model:**  A model (Maya) that shows improved performance in understanding cultural and linguistic nuances, especially in comparison to models trained primarily on English data. The results show that Maya performs comparably to or better than models of similar size (LLaVA-7B) and often approaches the performance of larger models (PALO-13B) on multilingual benchmarks. The toxicity filtering has a minimal impact on overall performance, suggesting that valuable information isn't lost by removing toxic content. The paper includes both quantitative benchmark results and qualitative examples demonstrating the model's capabilities.
</details>

### **MiniMax-01: Scaling Foundation Models with Lightning Attention**

MiniMax-01 is a series of large foundation models, including MiniMax-Text-01 and MiniMax-VL-01, that achieve performance comparable to top-tier models (like GPT-4o and Claude-3.5-Sonnet) while offering significantly longer context windows (up to 4 million tokens).  It achieves this through a novel architecture incorporating lightning attention (a highly efficient linear attention variant), Mixture of Experts (MoE), and optimized training/inference frameworks.

[![arXiv](https://img.shields.io/badge/arXiv-2501.08313-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.08313)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/MiniMax-AI/MiniMax-01)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/MiniMaxAI/MiniMax-VL-01)  
MiniMax, Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, Enwei Jiao, Gengxin Li, Guojun Zhang, Haohai Sun, Houze Dong, Jiadai Zhu, Jiaqi Zhuang, Jiayuan Song, Jin Zhu, Jingtao Han, Jingyang Li, Junbin Xie, Junhao Xu, Junjie Yan, Kaishun Zhang, Kecheng Xiao, Kexi Kang, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Zheng, Linbo Chai, Long Xing, Meizhi Ju, Mingyuan Chi, Mozhi Zhang, Peikai Huang, Pengcheng Niu, Pengfei Li, Pengyu Zhao, Qi Yang, Qidi Xu, Qiexiang Wang, Qin Wang, Qiuhui Li, Ruitao Leng, Shengmin Shi, Shuqi Yu, Sichen Li, Songquan Zhu, Tao Huang, Tianrun Liang, Weigao Sun, Weixuan Sun, Weiyu Cheng, Wenkai Li, Xiangjun Song, Xiao Su, Xiaodong Han, Xinjie Zhang, Xinzhu Hou, Xu Min, Xun Zou, Xuyang Shen, Yan Gong, Yingjie Zhu, Yipeng Zhou, Yiran Zhong, Yongyi Hu, Yuanxiang Fan, Yue Yu, Yufeng Yang, Yuhao Li, Yunan Huang, Yunji Li, Yunpeng Huang, Yunzhi Xu, Yuxin Mao, Zehan Li, Zekang Li, Zewei Tao, Zewen Ying, Zhaoyang Cong, Zhen Qin, Zhenhua Fan, Zhihang Yu, Zhuo Jiang, Zijia Wu

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Hybrid Attention:** The core innovation is the hybrid attention mechanism.  It primarily uses "lightning attention" (an I/O-aware implementation of TransNormer linear attention) for efficiency.  However, to maintain strong retrieval capabilities, it strategically inserts a standard transformer block with softmax attention after every seven transnormer blocks (with lightning attention).  This is a key differentiator from purely linear attention models, which often struggle with retrieval tasks. **Mixture of Experts (MoE):**  To scale the model efficiently, MiniMax-01 employs a Mixture of Experts (MoE) architecture in the feed-forward layers.  It has a massive 456 billion total parameters, but only 45.9 billion are activated for each token, using 32 experts with a top-2 routing strategy.  This allows for a large model capacity without a corresponding increase in computational cost per token. **Vision-Language Model (MiniMax-VL-01):** The vision-language model (MiniMax-VL-01) builds upon MiniMax-Text-01 by integrating a lightweight Vision Transformer (ViT) module.  It uses a dynamic resolution strategy, resizing input images to various sizes (from 336x336 to 2016x2016) and concatenating features from both resized patches and a standard thumbnail.  It *does not* use pooling or downsampling on the visual features, relying instead on the long-context capabilities of the architecture. Demonstrates the viability of linear attention at a massive scale, achieving performance comparable to top-tier models while significantly extending the context window. **Long-Context Capability:**  Supports context inputs of up to 4 million tokens, with strong performance in long-context evaluations. **Efficient Training and Inference Framework:**  Introduces several novel algorithmic and engineering optimizations to handle the hybrid architecture, MoE, and long contexts efficiently.
**Pre-training:** A curated corpus incorporating academic literature, books, web content, and programming code. **Vision-Language Pre-training (VL-01):** A substantial image-caption dataset (694 million unique pairs) and a dataset of 100 million images with fine-grained descriptions. **Vision-Language Instruction Data (VL-01):** A diverse instruction-based dataset synthesized from a wide array of image-related tasks. **Hybrid Attention:** The core fusion method combines the efficiency of lightning attention (linear) with the retrieval capabilities of softmax attention. **MoE Routing:** The top-2 routing strategy selectively activates experts, enhancing model capacity without increasing compute per token; a global router supports load balancing. **Vision-Language Fusion (VL-01):** Visual features from the ViT are projected into the LLM embedding space using a two-layer MLP. The raw, high-dimensional visual features are used without pooling or downsampling, leveraging the architecture's long-context capabilities. **Varlen Ring Attention and LASP+:** These algorithms enable efficient handling of long, variable-length sequences and data packing during training and inference.

</details>

### **NVLM: Open Frontier-Class Multimodal LLMs**

NVLM 1.0 is a family of multimodal large language models (LLMs) achieving state-of-the-art results on vision-language tasks, rivaling proprietary and open-access models. It demonstrates improved text-only performance after multimodal training and offers a comprehensive comparison of decoder-only and cross-attention-based architectures, introducing a novel hybrid architecture and a 1-D tile-tagging design for high-resolution images.

[![arXiv](https://img.shields.io/badge/arXiv-2409.11402-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2409.11402)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/NVIDIA/Megatron-LM/tree/NVLM-1.0)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/nvidia/NVLM-D-72B)  
Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping
<p align="center">
<img src="https://github.com/user-attachments/assets/da882643-ac1d-4566-8287-cd8da3897a88" width="600" alt="Architecture diagram for NVLM: Open Frontier-Class Multimodal LLMs" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**NVLM (NVIDIA Vision Language Model)** introduces a family of models with three primary architectures: NVLM-D (Decoder-only), NVLM-X (Cross-attention-based), and NVLM-H (Hybrid).  All models share a common vision pathway, employing a frozen InternViT-6B-448px-V1-5 vision encoder with dynamic high-resolution (DHR) processing.  DHR involves dividing input images into tiles (up to 6, with varying aspect ratios) and a downscaled global "thumbnail" tile.  These tiles are processed by the vision encoder, and the resulting 1024 tokens per tile are downsampled to 256 via pixel shuffling. **NVLM-D (Decoder-only):**  Connects the vision encoder to the LLM (Qwen2-72B-Instruct or Nous-Hermes-2-Yi-34B) via a 2-layer MLP projector.  It introduces a novel *1-D tile-tagging* design for handling high-resolution images. Text-based tile tags (e.g., `<tile_1>`) are inserted before the flattened image tokens of each tile to provide positional information to the LLM.  Training involves pretraining (frozen LLM and vision encoder, training only the MLP) and supervised fine-tuning (SFT) (unfrozen LLM and MLP).  Crucially, a high-quality text-only SFT dataset is included to maintain/improve text-only performance. **NVLM-X (Cross-attention-based):**  Uses gated cross-attention layers to process image tokens, similar to Flamingo, but *without* a Perceiver resampler.  Image features are projected to the LLM's hidden dimension with a one-layer MLP.  Gated X-attention layers are interleaved with LLM self-attention layers.  Training also has pretraining and SFT stages. The LLM backbone is unfrozen during SFT, and a high-quality text-only dataset is used. 1-D tile tags are also used, but within the X-attention layers. **NVLM-H (Hybrid):**  Combines aspects of NVLM-D and NVLM-X.  The thumbnail image tokens are processed by the LLM's self-attention layers (like NVLM-D), while the regular tile tokens are processed by gated cross-attention (like NVLM-X).  This aims to balance multimodal reasoning with computational efficiency. It also uses 1-D tile tags in cross-attention. The 1-D tile-tagging design significantly improves performance, especially on OCR-related tasks, compared to simply concatenating image tokens or using 2D grid/bounding box tags. The authors emphasize that dataset quality and task diversity are more important than sheer scale, even during pretraining. NVLM models achieve strong performance on *both* vision-language and text-only tasks.  This is achieved by including a high-quality text-only dataset during SFT and incorporating multimodal math and reasoning data. Decoder VS X-Attention: Cross attention based models are more efficient in high-resolution images. However, Decoder models provides unified multimodel reasoning and higher accuracy in OCR-related tasks. Curated from open-source datasets, including captioning (COCO, CC3M, SBU, LAION-115M), VQA (VQAv2, Visual Genome, DVQA), document understanding (Docmatix), OCR/Scene-Text (various datasets), and Math (CLEVR-Math).  Emphasis on quality over quantity. A diverse collection of task-oriented datasets, including captioning, VQA, chart/diagram understanding, document understanding, OCR, math, and science datasets. High-quality text-only data from various sources (ShareGPT, SlimOrca, EvolInstruct, etc.) and categories (general, math, coding) is crucial for maintaining/improving text-only performance.  Refined using GPT-40 and GPT-40-mini. NVLM models are evaluated on a wide range of vision-language benchmarks (MMMU, MathVista, OCRBench, AI2D, ChartQA, DocVQA, TextVQA, RealWorldQA, VQAv2) and text-only benchmarks (MMLU, GSM8K, MATH, HumanEval).
</details>

### **OmniVLM: A Token-Compressed, Sub-Billion-Parameter Vision-Language Model for Efficient On-Device Inference**

OmniVLM is a sub-billion-parameter vision-language model designed for efficient on-device inference, featuring a token compression mechanism that reduces visual token sequence length from 729 to 81, drastically cutting computational overhead while maintaining visual-semantic fidelity. It uses Qwen2.5-0.5B-Instruct model, Google's SigLIP-400M.

[![arXiv](https://img.shields.io/badge/arXiv-2412.11475-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.11475)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/NexaAIDev/OmniVLM-968M)  
Wei Chen, Zhiyuan Li, Shuo Xin
<p align="center">
<img src="https://github.com/user-attachments/assets/da2a140a-5efe-4499-addc-8ccbb3e9792a" width="600" alt="Architecture diagram for OmniVLM: A Token-Compressed, Sub-Billion-Parameter Vision-Language Model for Efficient On-Device Inference" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

OmniVLM addresses the challenges of deploying vision-language models (VLMs) on resource-constrained edge devices.  It achieves this through a novel token compression mechanism and a multi-stage training pipeline. The core innovation is the **image token compression**, which transforms the embedding dimensions from <code>&#91;batch_size, 729, hidden_size&#93;</code> to <code>&#91;batch_size, 81, hidden_size&#93;</code> within the projection layer. This 9x reduction in token count is achieved through reshaping, chosen after empirical comparison against convolution-based methods. The model architecture (Figure 1) builds upon the LLaVA framework, employing Google's SigLIP-400M as the vision encoder, Qwen2.5-0.5B-Instruct as the base language model, and a Multi-Layer Perceptron (MLP) as the projection layer.  The training pipeline consists of three stages: (1) **Pretraining** on large-scale image-caption pairs (primarily from the LLaVA pretraining dataset) to learn visual-linguistic alignments, training only the projection layer; (2) **Supervised Fine-Tuning (SFT)** on a mix of datasets (LLaVA, UnimmChat, and internal data) to improve contextual understanding and conversational coherence, training the projector and LLM while freezing the vision encoder; and (3) **Minimal-Edit Direct Preference Optimization (DPO)**, using a teacher model to create minimally edited corrections to the base model's outputs, forming chosen-rejected pairs for preference learning, again freezing the vision encoder and training the projector and LLM.  The DPO process leverages GPT-4V to generate synthetic training pairs. Extensive experiments show that the 81-token configuration provides the optimal balance between computational efficiency and model performance. OmniVLM outperforms nanoLLAVA on benchmarks like ScienceQA, POPE, and MMMU, demonstrating improved reasoning, multimodal comprehension, and generalization. Crucially, it achieves significantly faster inference speeds (9.1x faster time-to-first-token and 1.5x higher decoding speed compared to nanoLLAVA on a laptop, and 8x faster TTFT on a mobile device), making it suitable for deployment on edge devices like smartphones and laptops.
</details>

### **Pixtral 12B: A Cutting-Edge Open Multimodal Language Model**

Pixtral 12B is a 12-billion-parameter multimodal language model developed by Mistral AI, designed to excel in both understanding images and text, achieving leading performance on various multimodal benchmarks. The core of the VLM is built upon the transformer architecture. A strong aspect of the VLM is, Pixtral 12B is trained with a new vision encoder from scratch to natively support variable image sizes and aspect ratios.

[![arXiv](https://img.shields.io/badge/arXiv-2410.07073-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2410.07073)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pixtral.md)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/mistralai/Pixtral-12B-2409)  
Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, et al. (Mistral AI Science Team)
<p align="center">
<img src="https://github.com/user-attachments/assets/5187d3c0-e284-40eb-bb94-53105c8cbe11" width="600" alt="Architecture diagram for Pixtral 12B: A Cutting-Edge Open Multimodal Language Model" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Pixtral 12B** has two main components, *vision encoder (Pixtral-ViT)*, which tokenizes images and a *multimodal decoder*, which predicts the next token given a sequence of text and images. Pixtral can take an arbitrary number of images as an input, provided they fit within its 128K context window. **The vision encoder (Pixtral-ViT)** is trained from scratch with a novel ROPE-2D implementation, allowing it to process images at their native resolution and aspect ratio. The model can flexibly process images at low resolution in latency-constrained settings, while processing images at high resolution when fine-grained reasoning is required. For distinguishing between images with same number of patches but different aspect ratios, <code>&#91;IMAGE BREAK&#93;</code> tokens are inserted between image rows. Additionally, an <code>&#91;IMAGE END&#93;</code> token marks the end of image sequence. The model employs a **gated FFN** architecture, implementing gating in the hidden layer in place of standard feedforward layer in the attention block. For processing images within a single batch, the model flattens images along the sequence dimension and concatenates them. A block diagonal mask is constructed to prevent attention leakage between patches of different images. Traditional learned and absolute position embeddings are replaced by **ROPE-2D**, which allows handling variable image sizes. The **multimodal decoder** of Pixtral is built on top of Mistral Nemo 12B, a 12-billion parameter decoder-only language model. The decoder uses a causal self-attention. The vision encoder is connected to the multimodal decoder by a two-layer fully connected network. The paper describes Pixtral as an instruction-tuned model, pre-trained on large-scale interleaved image and text documents. The Paper contributes an open-source benchmark called **MM-MT-Bench**, for evaluating vision-language models. Pixtral excels at multimodal instruction following, surpassing comparable open-source models on the MM-MT-Bench benchmark.

The Pixtral pathway later scaled to **Pixtral Large** in November 2024 and became the visual component of Mistral's general-purpose multimodal releases. **Mistral Small 4**, released in 2026, folds Pixtral-style perception together with reasoning and agentic coding in one open model; because Mistral has not disclosed a distinct replacement vision architecture, it belongs to this lineage rather than a duplicate first-class entry. [Pixtral Large](https://mistral.ai/news/pixtral-large) · [Mistral Small 4](https://mistral.ai/news/mistral-small-4/)
</details>

### **Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos**

Sa2VA is a unified model for dense grounded understanding of both images and videos, integrating the SAM-2 video segmentation model with the LLaVA vision-language model. It supports a wide array of image and video tasks, like referring segmentation and conversation, by treating all inputs (text, images, videos) as tokens in a shared LLM space, generating instruction tokens that guide SAM-2 for precise mask production.

[![arXiv](https://img.shields.io/badge/arXiv-2501.04001-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.04001)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/magic-research/Sa2VA)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2501.04001)  
Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, Ming-Hsuan Yang

<p align="center">
<img src="https://github.com/user-attachments/assets/7527a503-4987-4401-961b-f52532788b1f" width="600" alt="Architecture diagram for Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Sa2VA leverages a pre-trained LLaVA-like model (containing a visual encoder, visual projection layer, and LLM) and appends SAM-2 alongside it.  Crucially, it uses a *decoupled design*, where SAM-2's decoder and memory module are frozen. This preserves SAM-2's perception and tracking capabilities and allows Sa2VA to be a plug-and-play module, updatable with newer MLLMs. The connection between the LLM and SAM-2 is a special <code>&#91;SEG&#93;</code> token. The LLM generates this token, and its hidden states act as a spatial-temporal prompt for SAM-2's decoder, which produces segmentation masks. The model is trained end-to-end, demonstrating scalability. The training uses a unified instruction-tuning format for various tasks: referring segmentation, visual question answering (VQA), and grounded conversation generation (GCG) for both images and videos. It treats all images, videos and prompts as visual tokens. A key aspect is the co-training with multiple datasets, including image and video data. The authors introduce *Ref-SAV*, an auto-labeled dataset with over 72,000 object expressions in complex video scenes, and manually validate 2,000 video objects in Ref-SAV for benchmarking referring video object segmentation. A simple mask tracking method re-utilizes SAM-2's knowledge. The model formulates all tasks as a single instruction-tuning process. Datasets used for co-training are: LLAVA 1.5 (665K), RefCOCO (17K), RefCOCO+ (17K), RefCOCOg (22K), Grand-f (214K), ChatUniVi (100K). Ref-YTVOS (3.5K), MeVIS (0.6K), ReVOS (1.7K) and Ref-SAV (37K).
</details>

### **Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding**

Tarsier2 is a state-of-the-art large vision-language model (LVLM) that excels in generating detailed and accurate video descriptions and demonstrates superior general video understanding capabilities. It scales pre-training data, performs fine-grained temporal alignment during supervised fine-tuning, and uses model-based sampling with Direct Preference Optimization (DPO) to improve performance, outperforming models like GPT-4o and Gemini 1.5 Pro.

[![arXiv](https://img.shields.io/badge/arXiv-2501.07888-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.07888) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/bytedance/tarsier) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/omni-research/Tarsier-7b)  
Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, Yuan Lin

<p align="center">
<img src="https://github.com/user-attachments/assets/e6626842-69ac-4547-8c4b-cb260dd349ca" width="600" alt="Architecture diagram for Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Tarsier2 utilizes a straightforward architecture consisting of a vision encoder, a vision adaptor, and a large language model (LLM), specifically building upon Qwen2-VL.  The model undergoes a three-stage training process: pre-training, supervised fine-tuning (SFT), and reinforcement learning (RL) using Direct Preference Optimization (DPO).  A key improvement over its predecessor, Tarsier, is the significant expansion of the pre-training dataset from 11 million to 40 million video-text pairs. This expansion includes the meticulous collection and filtering of 11 million commentary videos (explanations and analyses of movies and TV shows), providing rich contextual information.  During the SFT stage, Tarsier2 is trained on a dataset containing 150K instances, each with a detailed video description and specific frame annotations corresponding to each described event.  This *fine-grained temporal alignment* provides supervision that improves accuracy and reduces hallucinations compared to traditional video-caption alignment. The SFT phase is conducted in two steps. The initial step is frame to event allignment. Then, the model's output to make a more human-like style. The final training stage employs DPO with automatically generated preference data. Negative samples are created by corrupting videos (clip-switching, clip-reversing, clip-cropping, and down-sampling), and a preference data filtering method (using AutoDQ) ensures high-quality pairs.  Tarsier2 achieves state-of-the-art results on 15 public benchmarks, demonstrating its versatility across tasks such as video question-answering, video grounding, hallucination tests, and embodied question-answering.  A recaptioning dataset, Tarsier2-Recap-585K, is also released.
</details>

### **UI-TARS: Pioneering Automated GUI Interaction with Native Agents**

UI-TARS is a native GUI agent model that operates solely on screenshots, performing human-like interactions (keyboard and mouse operations). Unlike frameworks relying on wrapped commercial models (e.g., GPT-4o), UI-TARS is an end-to-end model achieving state-of-the-art (SOTA) performance on 10+ GUI agent benchmarks in perception, grounding, and task execution, significantly outperforming sophisticated frameworks.

[![arXiv](https://img.shields.io/badge/arXiv-2501.12326-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.12326) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/bytedance/UI-TARS) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/bytedance-research/UI-TARS-7B-SFT)  
Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, Guang Shi

<p align="center">
<img src="https://github.com/user-attachments/assets/9dccbdf3-a0ab-4ae4-930b-09a974f14a03" width="600" alt="Architecture diagram for UI-TARS: Pioneering Automated GUI Interaction with Native Agents" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

UI-TARS leverages several key innovations: (1) **Enhanced Perception**, utilizing a large-scale GUI screenshot dataset for context-aware understanding and precise captioning of UI elements; (2) **Unified Action Modeling**, standardizing actions into a unified space across platforms and achieving precise grounding through large-scale action traces; (3) **System-2 Reasoning**, incorporating deliberate reasoning for multi-step decision-making, including task decomposition, reflection, and milestone recognition; and (4) **Iterative Training with Reflective Online Traces**, addressing the data bottleneck by automatically collecting, filtering, and refining interaction traces on hundreds of virtual machines.  The model is trained iteratively and tuned via reflection, continuously learning from mistakes and adapting to new situations with minimal human intervention.  The architecture takes screenshots as input and uses a Vision-Language Model (VLM), specifically Qwen-2-VL 7B and 72B, to process visual information and generate actions.  The action space is unified across platforms (mobile, desktop, web) and includes actions like click, type, scroll, and drag.  Reasoning is infused by generating explicit "thoughts" before each action, inspired by reasoning-and-acting frameworks. These thoughts are generated through a combination of curated GUI tutorials and augmented action traces, incorporating patterns like task decomposition, long-term consistency, milestone recognition, trial and error, and reflection.  The training process involves multiple stages, starting with perception enhancement using a curated dataset of GUI screenshots and associated metadata.  This dataset supports tasks like element description, dense captioning, state transition captioning, question answering, and set-of-mark prompting. Action modeling is improved by creating a large-scale dataset of action traces and using grounding data to pair element descriptions with spatial coordinates.  The model is trained using a combination of supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) with reflection tuning to learn from errors.
</details>


### **VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling**

VideoChat-Flash is a system designed for handling long-form video content in multimodal large language models (MLLMs). It introduces a Hierarchical visual token Compression (HiCo) method to reduce computational load while preserving essential details, and uses a multi-stage learning approach with a new long-video dataset (LongVid) to achieve state-of-the-art performance on both long and short video benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2501.00574-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.00574) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenGVLab/VideoChat-Flash) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448)  
Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, Limin Wang

<p align="center">
<img src="https://github.com/user-attachments/assets/49048795-6a76-41e7-b441-1313d0813630" width="600" alt="Architecture diagram for VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Hierarchical visual token Compression (HiCo):**  This is the core innovation.  It compresses video information at two levels: **Clip-level Compression:**  The video is divided into clips.  A visual encoder (UMT-L) processes each clip, and a compressor (token merging with MLP) reduces the number of visual tokens.  This exploits inter-frame redundancy.  **Video-level Compression:** During the LLM (Qwen2-7B) interaction, visual tokens are further reduced using a progressive visual dropout strategy.  This leverages the idea that the LLM focuses on the entire context at shallow layers and specific details at deeper layers.  It combines uniform dropout (shallow layers) and text-guided selection (deep layers). **Visual Encoder:** UMT-L@224 (a video encoder, shown to be more efficient than image encoders like SigLIP). **Visual-Language Connector:**  A compressor (token merging) followed by an MLP projection. **Large Language Model (LLM):** Qwen2-7B. **Multi-stage Short-to-Long Learning:**  This is a crucial training strategy: **Stage 1: Video-Language Alignment:**  Train the compressor and MLP with image-text and short video-text pairs (0.5M each). **Stage 2: Short Video Pre-training:**  Enhance visual understanding with more images (3.5M) and short videos (2.5M). **Stage 3: Joint Short & Long Video Instruction Tuning:** Fine-tune on a mix of images (1.1M), short videos (1.7M), and long videos (0.7M) with instruction-following data. **Stage 4: Efficient High-Resolution Post-finetuning:**  Adapt to higher resolutions (224 to 448) by fine-tuning the video encoder on a subset (25%) of Stage 3 data.**Dynamic Video Sampling:**  Uses a dual sampling strategy: dense sampling (15 fps) for short videos (capturing motion) and sparse sampling (1 fps) for long videos (capturing events). **Timestamp-aware Prompt:** Uses a text prompt to provide timestamp information to the model. **LongVid:** A new large-scale long video instruction-tuning dataset introduced in the paper.  It contains 114,228 long videos and 3,444,849 question-answer pairs across five task types.  It leverages existing datasets (Ego4D, HowTo100M, HD-Vila, MiraData) and generates dense event labels. **Mixed Training Data:**  Uses a combination of short and long videos during training. **NIAH (Needle In A video Haystack)**. A newly created dataset for testing models capabilities for understanding long contexts.

</details>

### **VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding**

VideoLLaMA3 is a vision-centric multimodal foundation model designed for both image and video understanding, emphasizing a training paradigm and framework that prioritize high-quality image-text data, alongside an adaptable vision encoder and video token compression, to achieve state-of-the-art performance.

[![arXiv](https://img.shields.io/badge/arXiv-2501.13106-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.13106v1)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2501.13106)  
Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, Deli Zhao

<p align="center">
<img src="https://github.com/user-attachments/assets/350a1228-c14e-45ed-b59f-e99608ee9a7d" width=600 alt="Architecture diagram for VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**VideoLLaMA3** introduces a vision-centric approach in both its training paradigm and framework design, focusing on enhancing image and video understanding capabilities.  The core architecture incorporates a pre-trained vision encoder (SigLIP), a video compressor (DiffFP), a projector, and a large language model (LLM - Qwen2.5). The model employs a four-stage training process: 1) **Vision Encoder Adaptation**, where the vision encoder is adapted to accept images of variable resolutions using scene images, document data, and scene text images; 2) **Vision-Language Alignment**, which jointly tunes the vision encoder, projector, and LLM using large-scale image-text data (including detailed captions, documents, and charts) and a small amount of text-only data; 3) **Multi-task Fine-tuning**, incorporating image-text data for downstream tasks and general video caption data; and 4) **Video-centric Fine-tuning**, using general videos, streaming videos, temporally grounded videos, image-only, and text-only data. A key innovation is **Any-resolution Vision Tokenization (AVT)**, which allows the vision encoder to process images and videos of any resolution by replacing fixed positional embeddings with Rotary Position Embedding (RoPE).  This enables handling images with variable shapes and minimal information loss.  For video inputs, **Differential Frame Pruner (DiffFP)** acts as a video compressor, reducing the number of vision tokens by comparing the 1-norm distance between temporally consecutive patches in pixel space and pruning redundant patches.  This makes video representations more compact and precise. The training data mixture is carefully curated for each stage, emphasizing high-quality image-text data. The Vision Encoder Adaptation stage uses datasets like VL3-Syn7M-short, LLaVA-Pretrain-558k, and document datasets. The Vision-Language Alignment stage expands on this with detailed captions, OCR data, and fine-grained data with bounding boxes. The Multi-task Fine-tuning stage adds question-answering data and general video caption data. Finally, the Video-centric Fine-tuning stage includes general videos, streaming videos, and temporal grounding data. This "vision-centric" approach, prioritizing image understanding as a foundation for video understanding, along with AVT and DiffFP, allows VideoLLaMA3 to achieve strong performance on both image and video benchmarks.
</details>

### **Llama 3.2-Vision: Enhanced Multimodal Capabilities Built on Llama 3**

Llama 3.2-Vision extends the Llama 3 text-only model with multimodal capabilities, allowing it to process both text and images. This model, available in 11B and 90B parameter sizes, leverages a vision adapter with cross-attention layers to integrate image representations from a separate vision encoder into the core Llama 3 LLM, achieving strong performance on visual recognition, image reasoning, captioning, and visual question answering.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/meta-llama/llama-models) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)  
Meta
<p align="center">
<img src="https://github.com/user-attachments/assets/f6428237-8607-4de1-975d-dfc4c683b7a3" width=600 alt="Architecture diagram for Llama 3.2-Vision: Enhanced Multimodal Capabilities Built on Llama 3" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Llama 3.2-Vision** builds upon the Llama 3 architecture, an auto-regressive language model using an optimized transformer.  It adds a *vision adapter*, comprised of cross-attention layers, to incorporate visual information.  This adapter receives input from a *separate vision encoder* (not part of the core Llama 3 model), allowing the model to process images without directly converting them into text tokens. The `<|image|>` tag within the prompt signifies the presence of an image and dictates where the visual information is integrated via cross-attention. This integration occurs *after* the image tag and influences *subsequent* text tokens. The model supports a context length of 128k tokens and utilizes Grouped-Query Attention (GQA). The model family was trained on 6B image-text pairs. Pretraining data cutoff is December 2023, supports English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. However image-text tasks are only in English. The model's training involves supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) for instruction-tuned versions. The base models are suitable for text completion, with or without an image, using specific prompt formats.  Instruction-tuned models excel at tasks like Visual Question Answering (VQA), Document VQA (DocVQA), image captioning, and image-text retrieval. The training process includes stages of pretraining and annealing, leveraging a vast amount of data and significant computational resources (H100 GPUs). Key capabilities include handling both text and image inputs, answering questions about images, generating captions, and performing visual reasoning.  The model *does not* support built-in tool calling (like `brave_search` or `wolfram_alpha`) when an image is present in the prompt; tool calling is only available for text-only inputs.  The intended use cases cover a wide range of applications, but usage is restricted by the Llama 3.2 Community License and Acceptable Use Policy, particularly regarding languages and potential misuse. Meta emphasizes a responsible deployment approach, including providing tools like Llama Guard for safety and encouraging developers to implement appropriate safeguards. The model underwent extensive evaluations, including red teaming and assessments for critical risks such as CBRNE, child safety, and cyber attacks.
</details>

### **SmolVLM: A Small, Efficient, and Open-Source Vision-Language Model**

SmolVLM is a 2B parameter vision-language model (VLM) that achieves state-of-the-art performance for its memory footprint, offering a small, fast, and memory-efficient solution for multimodal tasks.  It is fully open-source, with all model checkpoints, datasets, training recipes, and tools released under the Apache 2.0 license, enabling local deployment, reduced inference costs, and user customization.

[![arXiv](https://img.shields.io/badge/Blog-SmolVLM%20Blog-b31b1b.svg?style=flat-square)](https://huggingface.co/blog/smolvlm)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/huggingface/smollm)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)  
Andres Marafioti, Merve Noyan, Miquel Farré, Elie Bakouch, Pedro Cuenca

<p align="center">
<img src="https://github.com/user-attachments/assets/901ed802-5c1c-4733-ab2a-6b61514b9c71" width="600" alt="Architecture diagram for SmolVLM: A Small, Efficient, and Open-Source Vision-Language Model" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

SmolVLM builds upon the architecture of Idefics3, leveraging a similar implementation in transformers but with key differences to enhance efficiency. It replaces the Llama 3.1 8B language backbone with the smaller SmolLM2 1.7B model.  A more aggressive image compression strategy is employed, using a pixel shuffle strategy that reduces visual information by a factor of 9 (compared to 4x in Idefics3).  This allows for 384x384 patches, and a shape-optimized SigLIP is used as the vision backbone with 14x14 inner patches. The model demonstrates superior memory usage compared to other VLMs in transformers, enabling efficient on-device inference.  For instance, encoding a single image and prompt requires only 1.2k tokens, significantly less than models like Qwen2-VL. This efficiency translates to faster prefill and generation throughputs. SmolVLM achieves strong performance on benchmarks such as MMMU, MathVista, MMStar, DocVQA, and TextVQA. It also shows promising results in basic video analysis, leveraging its long context capabilities. Training involved extending the context window of SmolLM2 to 16k tokens using techniques like RoPE base value adjustment and fine-tuning on a mixture of long- and short-context datasets. A curated training dataset, largely based on The Cauldron and Docmatix, was used for the VLM training. Checkpoint selection was based on a weighted metric across multiple vision-language benchmarks. The model is integrated with VLMEvalKit for easy evaluation, and it can be readily used and fine-tuned with the transformers library.  TRL integration allows for applying Direct Preference Optimization (DPO). A notebook is provided for fine-tuning on VQAv2, with options for LoRA, QLoRA, or full fine-tuning, even within the constraints of consumer GPUs.

Released on February 20, 2025, **SmolVLM2** extends the same architecture from still images to multi-image and video input while preserving edge-oriented 256M, 500M, and 2.2B scales. It adds temporal sampling and video instruction tuning to the Idefics3-derived visual-token pipeline rather than introducing a separate family. [Announcement](https://huggingface.co/blog/smolvlm2)
</details>

### **Idefics2**

IDEFICS2, an 8B parameter open-source vision-language model, efficiently processes interleaved image and text sequences by combining a SigLIP vision encoder, a Mistral-7B LLM, and a Perceiver pooling layer with MLP projection for robust text encoding, excelling in tasks like OCR and document understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2405.02246-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.02246) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/HuggingFaceM4/idefics-8b)  
Hugo Laurençon, Léo Tronchon, Matthieu Cord, Victor Sanh
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/c197c8c5-8da2-4d96-8999-8e05e81f1506" alt="Architecture diagram for Idefics2" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
IDEFICS2 is an 8B parameter open-source vision-language model adept at handling interleaved image and text sequences. IDEFICS2 utilizes a vision-language architecture designed for efficient processing of image and text sequences. It employs the SigLIP model as the vision encoder, extracting features from images in their native resolutions and aspect ratios. The Mistral-7B model serves as the LLM backbone, providing language understanding and generation capabilities. For text encoding, IDEFICS2 leverages a **Perceiver pooling layer** followed by an **MLP projection** to integrate visual features with the LLM's embedding space. This combination of vision encoder, LLM, and text encoder enables IDEFICS2 to handle various multimodal tasks, with a particular focus on OCR and document understanding.  The model is trained on a diverse dataset encompassing OBELICS, LAION Coco, and PMD, with additional data for OCR tasks. Fine-tuning is performed on instruction datasets like The Cauldron and OpenHermes-2.5.
</details> 

### **Idefics3-8B: Building and Better Understanding Vision-Language Models**

Idefics3-8B is a powerful open-source vision-language model (VLM) that significantly outperforms its predecessor, Idefics2-8B, while being trained efficiently and exclusively on open datasets. It leverages a straightforward pipeline and introduces Docmatix, a massive dataset for document understanding, to achieve state-of-the-art performance within its size category across various multimodal benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2408.12637-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.12637) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/HuggingFaceM4/idefics3)  
Hugo Laurençon, Andrés Marafioti, Victor Sanh, Léo Tronchon  
<p align="center">
<img src="https://github.com/user-attachments/assets/5e61fec2-b41b-4ad8-a167-1966f169b866" alt="Architecture diagram for Idefics3-8B: Building and Better Understanding Vision-Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Idefics3-8B builds upon the foundation of pre-trained unimodal models, specifically Llama 3.1 instruct as the language model and SigLIP-SO400M as the vision encoder. It adopts a self-attention architecture, where visual features are treated as tokens and concatenated with text tokens before being fed into the LLM. To enhance OCR capabilities and address the bottleneck of limited visual tokens per image, Idefics3-8B replaces the perceiver resampler used in Idefics2 with a simple pixel shuffle strategy, similar to InternVL-1.5. This strategy reduces the number of image hidden states by a factor of 4, allowing for the encoding of larger images (up to 364x364 pixels) into 169 visual tokens. The model utilizes an image-splitting strategy during both training and inference, dividing the original image into a matrix of 364x364 pixel tiles. To preserve the 2D structure and positional information of these tiles, a text token '\n' is inserted after each row of tiles, and the downscaled original image is appended to the sequence. Additionally, each tile is prepended with textual tokens indicating its position in the matrix. The training process consists of three stages of pre-training followed by supervised fine-tuning. In the first pre-training stage, the backbones (LLM and vision encoder) are frozen, and only the newly initialized parameters are trained. The maximum image resolution is gradually increased from 364² to 1820². From the second stage onward, the backbones are efficiently trained using DoRA (a variant of LoRA), and larger images are introduced into the training data. The final pre-training stage focuses on training with large synthetic datasets, including Docmatix, Websight, LNQA, PixelProse, and ChartGemma. During supervised fine-tuning, NEFTune noise is applied to the inputs, and the loss is calculated only on the answer tokens. The learning rate is kept constant for the first two pre-training stages and linearly decayed to zero during the final pre-training stage and supervised fine-tuning. Idefics3-8B demonstrates significant improvements over Idefics2, particularly in document understanding tasks, achieving a 13.7-point improvement on DocVQA. This highlights the effectiveness of the Docmatix dataset and the architectural choices made in Idefics3-8B. The model also achieves state-of-the-art performance within its size category across various multimodal benchmarks, including MMMU, MathVista, MMStar, and TextVQA, showcasing its strong capabilities in visual understanding and reasoning.
</details>

### **InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model**

InternLM-XComposer2 excels in free-form text-image composition and comprehension by connecting a CLIP pre-trained vision encoder with the powerful InternLM-2 LLM using a novel Partial LoRA module, enabling efficient alignment of visual and language tokens for enhanced multimodal understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2401.16420-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.16420) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/InternLM/InternLM-XComposer) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Willow123/InternLM-XComposer)  
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/732d3b7b-02de-42d3-ae76-800bf035b391" alt="Architecture diagram for InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**InternLM-XComposer2**: This model introduces a sophisticated architecture that leverages a vision encoder and a Large Language Model (LLM), interconnected through a Partial Low-Rank Adaptation (LoRA) module. This innovative setup allows InternLM-XComposer2 to effectively process both images and text, employing visual tokens generated by the vision encoder alongside language tokens derived from the tokenized text. The vision encoder, pre-trained using CLIP for image-language contrastive learning, and InternLM-2, which serves as the LLM with multi-lingual capabilities, are key components of this architecture. **The Partial LoRA** module distinguishes itself by aligning visual and language tokens through low-rank adaptation applied specifically to visual tokens, enhancing the model's multimodal understanding and processing efficiency. The training methodology of InternLM-XComposer2 is multifaceted, focusing on fine-tuning the vision encoder and Partial LoRA to align visual tokens with the LLM across various datasets. This process involves general semantic alignment, world knowledge alignment, and vision capability enhancement to refine the model's ability to interpret image information and compose text-image content. Supervised fine-tuning further includes multi-task training and free-form text-image composition, aiming to optimize the model's performance in leveraging image information for comprehensive text-image generation and understanding. Alignment techniques and fusion methods in InternLM-XComposer2 utilize the Partial LoRA module for the effective integration of different modalities, thereby enriching the LLM with modality-specific knowledge while preserving its inherent capabilities. This selective enhancement of visual tokens through Partial LoRA enables the model to exhibit robust performance across visual and textual domains, facilitating detailed perception, logical reasoning, and extensive knowledge integration in multimodal understanding. The model employs a diverse array of datasets, including ShareGPT4V-PT, COCO, Nocaps, TextCaps, and many others, for pre-training and supervised fine-tuning. These datasets serve to equip InternLM-XComposer2 with a broad range of capabilities, including general semantic alignment, world knowledge alignment, vision capability enhancement, and the facilitation of free-form text-image composition, marking a significant advancement in the field of vision-language large models.
</details> 

### **InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD**

InternLM-XComposer2-4KHD, building on its predecessor, pioneers high-resolution image handling in LVLMs by employing dynamic resolution with automatic patch configuration, adapting to resolutions from 336 pixels up to 4K HD for enhanced visual understanding without distortion.

[![arXiv](https://img.shields.io/badge/arXiv-2404.06512v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2404.06512v1)  
Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/c09b67fb-32eb-43de-82fa-96c3af22caf4" alt="Architecture diagram for InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>    
  
**InternLM-XComposer2-4KHD**: Cutting-edge Large Vision-Language Model (LVLM) designed to handle ultra-high resolutions, up to 4K HD and beyond, while also supporting diverse resolutions from 336 pixels. The model builds upon the InternLM-XComposer2 architecture, incorporating a novel **dynamic resolution with automatic patch configuration** technique. This allows the model to dynamically adjust patch layouts and counts based on the input image's aspect ratio, enabling efficient processing of high-resolution images while preserving their original proportions. To address potential ambiguity arising from variable patch configurations, a newline token is introduced to delineate rows of patch tokens, significantly improving performance. InternLM-XComposer2-4KHD is pre-trained on a diverse dataset, including image-caption pairs, concept knowledge, and OCR datasets, focusing on enhancing high-resolution and structural image understanding. Supervised fine-tuning further incorporates a mixed-resolution strategy, employing higher resolution for tasks requiring fine-grained detail, like HD-OCR tasks, and dynamically adjusted resolution for other tasks. This approach enables the model to excel in both high-resolution scenarios and general vision-language understanding tasks.
</details> 

### **InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output**

InternLM-XComposer-2.5 (IXC-2.5) is a versatile Large Vision Language Model (LVLM) designed to handle long-contextual input and output, excelling in various text-image comprehension and composition tasks. It achieves performance comparable to GPT-4V with a significantly smaller 7B LLM backend, demonstrating its efficiency and scalability.

[![arXiv](https://img.shields.io/badge/arXiv-2407.03320-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2407.03320) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/InternLM/InternLM-XComposer) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/Willow123/InternLM-XComposer)  
Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang  

<p align="center">
<img src="https://github.com/user-attachments/assets/1330a013-930b-4b23-90dc-94616b59ca0b" alt="Architecture diagram for InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

InternLM-XComposer-2.5 builds upon its previous iterations (IXC-2 and IXC-2-4KHD) and features a three-component architecture: a lightweight **OpenAI ViT-L/14 vision encoder**, a powerful InternLM2-7B LLM, and **Partial LoRA** for efficient alignment between the visual and language modalities. IXC-2.5 supports diverse input modalities, including text, single/multiple images, and videos. It utilizes a Unified Dynamic Image Partition strategy to handle high-resolution images and videos, resizing and padding them into smaller patches while preserving aspect ratios. For videos, frames are sampled and concatenated along the short side, creating a high-resolution composite image. The model is pre-trained in three stages: general semantic alignment, world knowledge alignment, and vision capability enhancement, using a diverse range of datasets. During pre-training, the LLM is frozen, and the vision encoder and Partial LoRA are fine-tuned to align visual tokens with the LLM. Supervised fine-tuning is then performed on a collection of datasets covering various tasks, including captioning, visual question answering, multi-turn QA, science QA, chart QA, math QA, OCR QA, video understanding, and conversation. This fine-tuning process involves jointly training all components with a weighted data sampling strategy and specific learning rate schedules for each component. IXC-2.5 also introduces two novel applications: crafting webpages and composing high-quality text-image articles. For webpage generation, the model is trained on a combination of synthetic and real-world web data, enabling it to generate HTML, CSS, and JavaScript code based on screenshots, instructions, or resume documents. For article composing, IXC-2.5 leverages Chain-of-Thought (CoT) and Direct Preference Optimization (DPO) techniques to enhance the quality of written content. This involves rewriting original prompts using CoT, generating diverse responses using different random seeds, and training a reward model to select preferred responses, ultimately leading to more creative and high-quality article generation.
</details>

### **InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling**

InternVL 2.5 is an advanced Multimodal Large Language Model (MLLM) series that builds upon InternVL 2.0, maintaining its core architecture while enhancing training and testing strategies, and data quality, to rival leading commercial models like GPT-4o and Claude-3.5-Sonnet.

[![arXiv](https://img.shields.io/badge/arXiv-2412.05271-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.05271)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenGVLab/InternVL)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/OpenGVLab/InternVL2_5-78B)  
Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang
<p align="center">
<img src="https://github.com/user-attachments/assets/d1651bde-a587-4b60-83e4-40468d6442ee" width="600" alt="Architecture diagram for InternVL 2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**InternVL 2.5** retains the "ViT-MLP-LLM" architecture of its predecessors, combining a pre-trained InternViT (either InternViT-6B or InternViT-300M) with LLMs of varying sizes (InternLM 2.5, Qwen 2.5) via a 2-layer MLP projector. A key feature is the pixel unshuffle operation, reducing visual tokens from 1024 to 256 per 448x448 image tile, improving scalability for high-resolution processing. The model architecture supports dynamic resolution, adapting to image aspect ratios by dividing images into 448x448 tiles.  Crucially, InternVL 2.0 and 2.5 incorporate multi-image and video data, in addition to single-image and text-only data. The training strategy involves a three-stage pipeline:  (1) MLP warmup, where only the MLP projector is trained, (2) optional ViT incremental learning, where the vision encoder and MLP are trained to enhance visual feature extraction, particularly for domains rare in web-scale data, and (3) full model instruction tuning, where the entire model is trained on high-quality multimodal instruction datasets.  A progressive scaling strategy is employed, starting with smaller LLMs and scaling up, allowing for efficient alignment of the vision encoder with larger LLMs.  Training enhancements include random JPEG compression (for robustness to real-world image quality) and loss reweighting (to balance contributions from responses of different lengths). Data organization is optimized using parameters like `nmax` (maximum tile number) and a repeat factor (`r`) to control data sampling frequency. A data-packing strategy concatenates multiple samples into longer sequences to improve GPU utilization. A significant contribution is a data filtering pipeline to remove low-quality samples, particularly those with repetitive patterns, mitigating the risk of repetitive generation, a common issue in MLLMs. The data mixture includes a wide range of tasks (captioning, general QA, mathematics, charts, OCR, etc.) and modalities (single-image, multi-image, video, text). The model was evaluated comprehensively on diverse benchmarks including multi-discipline reasoning (MMMU, MMMU-Pro), document understanding (DocVQA), multi-image/video understanding, real-world comprehension, multimodal hallucination detection, visual grounding, multilingual capabilities, and pure language processing.
</details>

### **DeepSeek-VL: Towards Real-World Vision-Language Understanding**

DeepSeek-VL, utilizing a hybrid vision encoder combining SigLIP-L and SAM-B, excels in real-world vision-language understanding by efficiently processing high-resolution images and integrating extracted features with a DeepSeek LLM backbone through a two-layer hybrid MLP adapter.

[![arXiv](https://img.shields.io/badge/arXiv-2403.05525-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2403.05525) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/deepseek-ai/DeepSeek-VL)

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, Chong Ruan  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/7b7283d2-b2d5-4ab6-891a-18a9760ef7ca" alt="Architecture diagram for DeepSeek-VL: Towards Real-World Vision-Language Understanding" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  

**DeepSeek-VL**: Employs a hybrid vision encoder architecture, fusing a **SigLIP-L encoder** for semantic understanding with a **SAM-B encoder** for high-resolution detail extraction. This allows for efficient processing of 1024x1024 images while capturing both global and fine-grained visual features. **A two-layer hybrid MLP adapter** then integrates these features with the DeepSeek LLM backbone. The model is pre-trained on a diverse dataset encompassing web screenshots, PDFs, OCR, charts, and knowledge-based content from sources like Common Crawl, Web Code, E-books, and arXiv articles. This pretraining is further refined using a curated instruction-tuning dataset based on real user scenarios and categorized into a comprehensive taxonomy covering recognition, conversion, analysis, reasoning, evaluation, and safety tasks. By combining this diverse data with its unique architecture and fusion strategies, DeepSeek-VL aims to deliver robust performance across a wide range of real-world vision-language applications.  
</details> 

### **DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding**

DeepSeek-VL2 is an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL, by incorporating a dynamic tiling vision encoding strategy for high-resolution images and leveraging DeepSeekMoE models with Multi-head Latent Attention for efficient inference. It is trained on a large vision-language dataset, shows top performance in tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2412.10302-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2412.10302)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/deepseek-ai/DeepSeek-VL2)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small)  
Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, and et al.

<p align="center">
<img src="https://github.com/user-attachments/assets/6bf7a0ce-5fa1-46ae-9f24-cb75df607a19" alt="Architecture diagram for DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

DeepSeek-VL2 builds upon a LLaVA-style architecture. it consists of three core modules: (1) a vision encoder, (2) a vision-language adaptor, and (3) a Mixture-of-Experts language model. It introduces two major enhancements: a dynamic tiling strategy and it uses DeepSeekMOE language model that has Multi-head Latent Attention. The dynamic tiling strategy addresses the limitations of fixed-resolution encoders by splitting high-resolution images into tiles. It uses a single SigLIP-SO400M-384 vision encoder. A set of candidate resolutions CR = {(m· 384, η · 384) | m∈ N, n ∈ N, 1 ≤ m, n,mn ≤ 9} is defined, representing different aspect ratios. For an input image, the optimal resolution from CR that minimizes padding is selected. The resized image is then divided into m₁ × n₁ local tiles of 384 × 384 pixels, plus one global thumbnail tile. The SigLIP-SO400M-384 processes all (1 + m¡ × n₁) tiles, yielding 729 visual embeddings (27x27) of 1152 dimensions per tile. Dynamic tiling is disabled for multiple (>2) images for efficiency. A 2x2 pixel shuffle compresses each tile's visual tokens to 14x14 (196 tokens). Special tokens are added: 14 <tile_newline> tokens at the end of each row in the global thumbnail (total 210 tokens); m₁ · 14 <tile_newline> tokens at the end of the final column of the local tiles; and a <view_separator> token between the global thumbnail and local tiles. The total visual sequence length is 210 + 1 + m₁ · 14 × (nį · 14 + 1). This sequence is projected into the LLM's embedding space by a two-layer MLP. The language model utilizes DeepSeekMoE, featuring Multi-head Latent Attention (MLA) to compress the Key-Value (KV) cache, improving inference speed and throughput. The MoE architecture further enhances efficiency. A global bias term is used during MoE training for load balancing. DeepSeek-VL2 comes in three variants (Tiny, Small, and Base) with 1.0B, 2.8B, and 4.5B activated parameters, respectively. The training data is constructed in three stages: (1) VL alignment, (2) VL pretraining, and (3) supervised fine-tuning (SFT). The alignment stage uses ShareGPT4V (1.2M samples). Pretraining data combines VL and text-only data (70/30 ratio), including interleaved image-text data (WIT, WikiHow, OBELICS, Wanjuan, and in-house data), image captioning data (various open-source datasets with quality enhancements and filtering), OCR data (LaTeX OCR, 12M RenderedText, and in-house data), general VQA data, table/chart/document understanding data (PubTabNet, FinTabNet, Docmatix), web-to-code and plot-to-Python data (Websight, and Python plots), QA with visual prompts, visual grounding data and grounded conversation data. SFT data includes enhanced general visual question-answering data, cleaned OCR and document understanding data, enhanced table and chart understanding data, improved reasoning/logic/math data, textbook/academic questions, and expanded web-to-code and plot-to-Python data, visual grounding data, grounded conversation data. Text only datasets were used during SFT stage. The training methodology involves a three-stage pipeline. Stage 1 trains the vision encoder and vision-language adaptor MLP, keeping the language model fixed, using image-text paired data. Stage 2 performs vision-language pre-training with all parameters unlocked, using ~800B image-text tokens. Stage 3 conducts supervised fine-tuning. Visual understanding is emphasized, and the loss is computed only on text tokens. Unlike previous work, the fixed-resolution vision encoder is adapted during training.

</details>


### **MANTIS: Mastering Multi-Image Understanding Through Interleaved Instruction Tuning**

MANTIS is a family of open-source large multimodal models that demonstrate state-of-the-art performance on multi-image visual language tasks. By focusing on instruction tuning with a carefully curated multi-image dataset, MANTIS achieves superior results using significantly less data than models trained with massive web datasets. This efficient approach opens new avenues for developing powerful multi-image LMMs with limited resources.

[![arXiv](https://img.shields.io/badge/arXiv-2405.01483-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.01483) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/TIGER-AI-Lab/Mantis) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TIGER-Lab/Mantis)  
Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, Wenhu Chen  
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/dd4bbdf4-5ab9-4e12-89bd-94c5beb2d114" alt="Architecture diagram for MANTIS: Mastering Multi-Image Understanding Through Interleaved Instruction Tuning" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary> 

**Mantis**: a powerful and efficient multi-image Large Multimodal Models (LMMs), demonstrating that massive pre-training on noisy web data is not the only path towards achieving state-of-the-art performance in complex visual-language tasks. Instead, MANTIS focuses on instruction tuning using high-quality, academic-level data, achieving remarkable results on various multi-image benchmarks while using significantly less data than its counterparts. Central to MANTIS's success is the meticulously curated MANTIS-INSTRUCT dataset, a collection of 721K multi-image instruction data carefully designed to instill four crucial skills: co-reference, comparison, reasoning, and temporal understanding. These skills equip MANTIS with a comprehensive toolkit for tackling the challenges of multi-image understanding. Co-reference enables the model to understand references like "second image" in natural language and correctly identify the corresponding image within the input. Comparison fosters the ability to analyze and identify subtle differences and commonalities between multiple images, a skill crucial for tasks like visual similarity assessment and difference description. Reasoning empowers the model to go beyond simple comparisons and make complex inferences by combining its world knowledge with the information extracted from multiple images, allowing it to solve intricate logical reasoning puzzles and answer challenging multi-image questions. Finally, temporal understanding equips MANTIS with the capability to process and understand image sequences, capturing the dynamic aspects of videos, comics, and other temporal visual data. MANTIS leverages a simple yet effective architecture based on existing pre-trained LLMs like LLaMA-3 and vision transformer encoders from CLIP or SigLIP. A multimodal projector, similar to the one used in LLaVA, aligns the visual embeddings with the text embeddings, facilitating their seamless integration within the LLM. This streamlined approach avoids the complexity of previous architectures like Q-Former while retaining high performance. Extensive evaluations on five multi-image benchmarks, including NLVR2, QBench, BLINK, MVBench, and a newly curated Mantis-Eval dataset, demonstrate MANTIS's superior performance, exceeding existing open-source LMMs and even matching the results of the powerful GPT-4V. Notably, MANTIS surpasses Idefics2-8B, a model pre-trained on 200x larger interleaved multi-image data, showcasing the effectiveness of instruction tuning with high-quality academic-level data. Furthermore, MANTIS retains strong single-image performance on par with existing state-of-the-art models, demonstrating its versatility and adaptability. MANTIS's impressive results, combined with its efficient training and open-source nature, offer a compelling alternative to traditional pre-training-heavy approaches, opening new possibilities for researchers and practitioners seeking to develop powerful and versatile multi-image LMMs with minimal computational resources.
</details> 

### **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**

Qwen-VL distinguishes itself by integrating a Vision Transformer with a large language model through a novel vision-language adapter, employing cross-attention mechanisms for precise alignment of visual and linguistic data, achieving high performance in various vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2308.12966-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.12966) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/qwenlm/qwen-vl)

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c9358aad-63e2-44d3-b3af-38e9d4f6aeaa" alt="Architecture diagram for Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Qwen-VL**: Represents an advanced architecture in the vision-language domain, constructed on a foundational large language model with the integration of a Vision Transformer (ViT) for visual encoding. This model stands out for its innovative approach to processing and aligning visual and linguistic data, featuring a **vision-language adapter equipped with cross-attention mechanisms**. These mechanisms enable the efficient compression and integration of image features into the language model, a critical component for achieving precise alignment between visual inputs and text. The architecture's design focuses on optimizing the handling of image features, employing a position-aware strategy to maintain spatial relevance of visual data when merged with textual information.The training methodology of Qwen-VL is meticulously structured into **three distinct phases**, starting with an **initial pre-training** on a diverse collection of weakly labeled image-text pairs. This is followed by **multi-task pre-training**, utilizing high-quality annotated datasets and larger input resolutions to refine the model's capabilities in various tasks such as instruction following and dialogue. The final phase involves **supervised fine-tuning**, further honing the model's performance across a spectrum of vision-language tasks. Special tokens and bounding box inputs are utilized for differentiating between image and text inputs and achieving fine-grained visual understanding, respectively.Qwen-VL's alignment techniques are innovative, employing a cross-attention mechanism within its vision-language adapter to fuse visual and textual features effectively. This approach ensures the preservation of spatial information post feature compression through the use of positional encodings. The model leverages an extensive suite of datasets for training, including LAION-en, LAION-zh, and various others for pre-training, alongside specialized datasets like GQA, VGQA, and VQAv2 for multi-task pre-training. These datasets are instrumental in supporting a broad array of vision-language tasks, emphasizing multilingual capabilities, fine-grained visual understanding, and the model's proficiency in captioning, visual question answering, grounding, and OCR tasks.
</details> 

### **Qwen2-VL: A Powerful Open-Source Vision-Language Model for Image and Video Understanding**

Qwen2-VL is the latest iteration of the Qwen vision-language model family, building upon the Qwen-VL architecture and introducing significant enhancements for improved understanding of images and videos. It excels in various tasks, including visual question answering, dialogue, content creation, and even agent-based control of devices like mobile phones and robots.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen2-VL) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)  
Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren  

<p align="center">
<img src="https://github.com/user-attachments/assets/37c2fb7a-66e1-475f-86e4-f00b4ac1c879" alt="Architecture diagram for Qwen2-VL: A Powerful Open-Source Vision-Language Model for Image and Video Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Qwen2-VL continues to leverage the core architecture of Qwen-VL, combining a Vision Transformer (ViT) with approximately 600M parameters and Qwen2 language models. This ViT is designed to handle both image and video inputs seamlessly. The key architectural improvements in Qwen2-VL include Naive Dynamic Resolution support and Multimodal Rotary Position Embedding (M-ROPE). Naive Dynamic Resolution allows the model to handle arbitrary image resolutions by mapping them into a dynamic number of visual tokens. This ensures that the model input accurately reflects the information content of the image, regardless of its original resolution. This approach is more aligned with human visual perception, which adapts to different image sizes and resolutions. M-ROPE enhances the model's ability to capture positional information in multimodal inputs. It deconstructs the original rotary embedding into three parts, representing temporal, height, and width information. This allows the LLM to simultaneously process and integrate 1D textual, 2D visual (image), and 3D video positional information, leading to a more comprehensive understanding of the input sequence. These architectural enhancements, combined with a robust training process, enable Qwen2-VL to achieve state-of-the-art performance on various visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, and MTVQA. It can also understand videos over 20 minutes long, enabling high-quality video-based question answering, dialogue, and content creation. Furthermore, Qwen2-VL's capabilities in complex reasoning and decision-making allow it to be integrated with devices like mobile phones and robots for automatic operation based on visual input and text instructions. The model also supports multilingual understanding of text within images, including most European languages, Japanese, Korean, Arabic, and Vietnamese, broadening its applicability to a global user base.
</details>

### **Qwen2.5-VL: Enhanced Vision-Language Capabilities in the Qwen Series**

Qwen2.5-VL represents a significant advancement in the Qwen series of vision-language models, offering improved image recognition, precise object grounding, enhanced text recognition, document parsing, and video comprehension, while also functioning as a visual agent capable of computer and phone use.

[![arXiv](https://img.shields.io/badge/Blog-Qwen%20Team%20Blog-b31b1b.svg?style=flat-square)](https://qwenlm.github.io/blog/qwen2.5-vl/)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/QwenLM/Qwen2.5-VL)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)  
Qwen Team

<p align="center">
<img src="https://github.com/user-attachments/assets/59f0878d-42c1-4013-af78-406b2f4763fe" width=600 alt="Architecture diagram for Qwen2.5-VL: Enhanced Vision-Language Capabilities in the Qwen Series" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Qwen2.5-VL builds upon its predecessor, Qwen2-VL, with substantial improvements in perception of temporal and spatial scales, as well as a simplified network structure for increased efficiency. **World-wide Image Recognition:**  Expanded recognition capabilities covering a vast array of categories, including landmarks, objects, and even film/TV IPs. **Precise Object Grounding:**  Uses bounding boxes and point-based representations for object localization, with standardized JSON output for coordinates and attributes, enabling hierarchical positioning. **Enhanced Text Recognition (OCR):**  Improved multi-scenario, multi-language, and multi-orientation text recognition and localization, with enhanced information extraction for applications like document processing. **Powerful Document Parsing:**  Introduces "QwenVL HTML" format, leveraging HTML for layout information extraction from documents, magazines, research papers, web pages, and mobile screenshots. **Enhanced Video Comprehension:** Supports understanding of ultra-long videos (hourly scale) with dynamic frame rate (FPS) training and absolute time encoding.  Enables second-level event localization and key point summarization. **Visual Agent Capabilities:** Can function as a visual agent for computer and phone use, capable of reasoning and dynamically directing tools. Capable of tasks like booking flights. **Time and Image Size Perception** In the spatial dimension, the model is capable of adapting varying image sizes into tokens and directly represents coordinates by detection boxes. In the temporal dimension, the model can comprehend the pace of time through temporal dimension. **Visual Encoder** A native dynamic resolution ViT is trained from scratch. Window Attention is used to minimize computational load. The model comes in three sizes (3B, 7B, and 72B parameters), with both base and instruct-tuned versions available.  The 72B-Instruct model achieves competitive performance on various benchmarks, excelling in document and diagram understanding.  Smaller models also demonstrate strong performance, with the 7B-Instruct model outperforming GPT-4o-mini in several tasks and the 3B model exceeding the performance of the previous Qwen2-VL 7B model. The models is trained with 18 trillion tokens. Future developments aim to further enhance problem-solving, reasoning, and multi-modality integration.
</details>

### **moondream1 and moondream2**

moondream1 and moondream2 are vision-language models with moondream2 building upon moondream1's SigLIP vision encoder and Phi-1.5 language backbone by incorporating an MLP projector for enhanced visual and textual representation alignment.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vikhyat/moondream) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vikhyatk/moondream2)  
@vikhyatk
<p align="center">
<img src="https://github.com/gokayfem/awesome-vlm-architectures/assets/88277926/e979d327-3423-4a91-92f2-02a3dc3189a8" alt="Architecture diagram for moondream1 and moondream2" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
  
**moondream1 and moondream2**: A series of vision-language models. moondream1 is a 1.6B parameter model that leverages **SigLIP** as the vision encoder and **Phi-1.5** as the language backbone, trained on the LLaVA dataset. moondream2 expands upon this foundation, utilizing a 1.86B parameter model initialized with weights from SigLIP and Phi-1.5. It incorporates **an MLP projector** to bridge the visual and textual representations, potentially leading to enhanced vision-language alignment and improved performance across various tasks.
</details>

### **Moondream-next: Compact Vision-Language Model with Enhanced Capabilities**

Moondream is a compact (1.9B parameters) vision-language model (VLM) that prioritizes practical usability and accessibility, offering features like structured output (JSON, XML, Markdown, CSV), improved OCR, and a novel experimental Gaze Detection capability, while maintaining fast performance and ease of deployment.

[![arXiv](https://img.shields.io/badge/Blog-Moondream%20Blog-b31b1b.svg?style=flat-square)](https://moondream.ai/)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vikhyat/moondream)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/vikhyatk/moondream-next)


<details>
<summary>ℹ️ <i>More Information</i></summary>

Moondream distinguishes itself by being exceptionally small (1.9B parameters) while supporting a wide range of functionalities typically found in larger, more specialized models. The architecture is not explicitly detailed in the provided text, but it mentions improvements to the "vision layer" for better OCR performance.  This suggests a structure where visual input is processed by a vision encoder, and then integrated with a language model.  The key feature is its ability to perform multiple Vision AI tasks ("capabilities") within a single, unified model, including: object detection, captioning, visual querying, pointing (x,y coordinate retrieval), and the newly added gaze detection.  The model also newly supports structured output formats, generating outputs directly as JSON, XML, Markdown, or CSV, making integration with applications much easier. The "Gaze Detection" capability is specifically highlighted as a novel and experimental feature, indicating a focus on real-world applications beyond standard benchmarks.  The training data and process are not thoroughly described, although the text notes increased training on "document querying and understanding" for OCR enhancement. The model's creators express a cautious approach to benchmarks, acknowledging their limitations and potential for manipulation, yet also highlight improved benchmark scores in this release, suggesting a balance between practical utility and measurable performance. It does not rely on external apis.
</details>

### **SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models**

SPHINX-X refines multi-modal large language models by streamlining its architecture to use two visual encoders, CLIP-ConvNeXt and DINOv2, and implementing an efficient single-stage training process for enhanced performance across diverse multi-modal tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2402.05935-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2402.05935) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alpha-vllm/llama2-accessory) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/Alpha-VLLM/SPHINX)  
Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, Wenqi Shao, Chao Xu, Conghui He, Junjun He, Hao Shao, Pan Lu, Hongsheng Li, Yu Qiao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/1c4e9a86-9a21-4911-bcb6-d2a79c181510" alt="Architecture diagram for SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**SPHINX-X**: Represents an advanced iteration in the development of Multi-modal Large Language Models (MLLM), building upon its predecessor, SPHINX, by optimizing both architecture and training efficiency. The core modifications introduced in SPHINX-X include the elimination of redundant visual encoders, the incorporation of **learnable skip tokens** to bypass **fully-padded sub-images**, and the simplification of the multi-stage training process into a singular, **all-in-one training** paradigm. This approach is designed to enhance the model's efficiency and effectiveness across a broad spectrum of multi-modal tasks. The architecture of SPHINX-X retains two key visual encoders, **CLIP-ConvNeXt and DINOv2**, ensuring robust text-image alignment capabilities, especially for high-resolution images and varied aspect ratios. This streamlined model architecture enables a unified encoding approach for both vision and text, emphasizing scalable and efficient training methodologies. The training strategy is comprehensive, directly engaging all model parameters across a wide-ranging multi-modal dataset, which encompasses public resources covering language, vision, and vision-language tasks. Additionally, SPHINX-X enriches this dataset with specially curated OCR-intensive and Set-of-Mark datasets to further extend the model's versatility and generalization capabilities. The datasets utilized in SPHINX-X aim to foster a deep, comprehensive understanding across multiple domains, enhancing the model's performance in OCR, document layout detection, and fine-grained multi-modal understanding. By training over various base Large Language Models (LLMs) with different parameter sizes and multilingual capabilities, SPHINX-X achieves a spectrum of MLLMs that showcase a strong correlation between multi-modal performance and the scales of data and parameters involved. This strategy allows SPHINX-X to set a new benchmark in multi-modal large language model performance, significantly advancing the field's capabilities in handling complex, multi-domain tasks.
</details>

### **BLIP: Bootstrapping Language-Image Pre-training**

BLIP introduces a versatile Multimodal Mixture of Encoder-Decoder (MED) architecture, integrating a visual transformer and a BERT-based text encoder with cross-attention layers, enabling unified vision-language understanding and generation across a wide range of tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2201.12086) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/BLIP)  
Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/27db1037-2b48-4097-9891-019ba77fc536" alt="Architecture diagram for BLIP: Bootstrapping Language-Image Pre-training" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIP**: Introduces an innovative approach to unified vision-language understanding and generation through its Multimodal Mixture of Encoder-Decoder (MED) architecture. This architecture is designed to be highly versatile, capable of serving as a unimodal encoder, an image-grounded text encoder, or an image-grounded text decoder. This flexibility allows BLIP to adeptly handle a wide array of vision-language tasks, showcasing its adaptability across various applications. The MED architecture incorporates a Visual Transformer to encode images, a BERT-based text encoder for processing textual information, additional **cross-attention layers** to facilitate image-text interaction, and **causal self-attention layers** for generating text based on image inputs. These components enable BLIP to support three key functionalities: encoding of either modality on its own, encoding of text grounded in images, and decoding of text from images, thus covering a comprehensive range of tasks from understanding to generation.BLIP's training methodology is grounded in the joint optimization of three pre-training objectives: Image-Text Contrastive Learning (ITC), Image-Text Matching (ITM), and Image-Conditioned Language Modeling (LM). These objectives are designed to align visual and textual features, learn fine-grained image-text alignment, and enable text generation from images, respectively. The model utilizes a mix of human-annotated and web-collected noisy image-text pairs for training, balancing the precision of manually annotated data with the scale and diversity of data collected from the web. This approach ensures robustness and scalability in BLIP's performance across vision-language tasks.For alignment and fusion of multimodal information, BLIP employs ITC and ITM losses to achieve precise text-image alignment, utilizing a multimodal representation that accurately captures the nuanced relationship between visual and textual data. The architecture's cross-attention layers play a crucial role in incorporating visual information into the text encoder for image-grounded text encoding. Simultaneously, modifications to the self-attention layers in the decoder facilitate text generation, effectively merging vision and text for unified processing. BLIP's pre-training leverages a diverse set of datasets, including COCO, Visual Genome, Conceptual Captions, Conceptual 12M, SBU Captions, and LAION. These datasets are instrumental in learning a broad spectrum of vision-language tasks, with high-quality human-annotated pairs and extensive web datasets providing the necessary depth and breadth for comprehensive pre-training.
</details> 

### **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

BLIP-2 leverages the power of frozen pre-trained image encoders and large language models, connecting them through a lightweight Querying Transformer (Q-Former) to efficiently extract and integrate visual features for enhanced vision-language understanding and generation.

[![arXiv](https://img.shields.io/badge/arXiv-2301.12597-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2301.12597) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Salesforce/BLIP2)  
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/604460f9-478c-4cc1-ba35-287447c04b26" alt="Architecture diagram for BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" />
</p>  
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIP-2**: The model architecture integrates frozen pre-trained image encoders and large language models (LLMs), employing a lightweight **Querying Transformer (Q-Former)** to facilitate the interaction between these modalities. The Q-Former plays a crucial role in extracting and integrating visual features relevant to textual queries, allowing for a more nuanced understanding and generation of language based on visual inputs.The training methodology of BLIP-2 is structured around a two-stage pre-training strategy. Initially, it focuses on learning vision-language representations utilizing the frozen image encoders. Subsequently, it advances to vision-to-language generative learning, leveraging the capabilities of frozen LLMs. This strategy, coupled with the use of **learnable query vectors within the Q-Former**, enables effective vision-language alignment. The alignment process is further enhanced through fusion methods that extract language-informative visual representations, which are then synthesized with the outputs of LLMs to generate pertinent textual descriptions. A diverse array of datasets including COCO, Visual Genome, CC3M, CC12M, SBU, and LAION400M underpins the comprehensive pre-training regime of BLIP-2. These datasets provide a rich variety of image-text pairs, essential for training the model across a broad spectrum of visual representations and language generation tasks. The model's architecture and training approaches are designed to address the prohibitive costs associated with vision-and-language pre-training, offering a more efficient pathway to developing multimodal understanding and generation capabilities.
</details> 

### **xGen-MM (BLIP-3): An Open-Source Framework for Building Powerful and Responsible Large Multimodal Models**

xGen-MM (BLIP-3) is a comprehensive framework developed by Salesforce for training a series of open-source large multimodal models (LMMs) designed to excel in a variety of visual language tasks. It provides meticulously curated datasets, a streamlined training recipe, model architectures, and a suite of open LMMs capable of performing various visual language tasks. xGen-MM focuses on scalability, using a simplified architecture and a unified training objective to enable training on larger, more diverse datasets. The framework also includes a safety-tuned model to mitigate harmful behaviors and promote responsible AI development.

[![arXiv](https://img.shields.io/badge/arXiv-2408.08872-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.08872) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/Salesforce/xgen-mm-1-models-and-datasets-662971d6cecbf3a7f80ecc2e)  
Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, Shrikant Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, Ran Xu  

<p align="center">
<img src="https://github.com/user-attachments/assets/e6e166c8-871e-420c-bbf1-b64c3c22e06a" alt="Architecture diagram for xGen-MM (BLIP-3): An Open-Source Framework for Building Powerful and Responsible Large Multimodal Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

xGen-MM (BLIP-3), short for xGen-MultiModal, addresses limitations of previous open-source efforts by providing a complete ecosystem for LMM development. Central to its approach is the utilization of diverse, large-scale, and high-quality multimodal data, which enables xGen-MM to achieve competitive performance against both open-source and proprietary LMMs. Instead of relying on the intricate Q-Former architecture and multiple training objectives used in its predecessor, BLIP-2, xGen-MM streamlines the process by employing a more scalable vision token sampler (perceiver resampler) and unifying the training objective to a single auto-regressive loss on text tokens. This simplification enables larger-scale training and focuses the model on effectively learning from the rich multimodal context. Furthermore, xGen-MM incorporates safety measures, introducing a safety-tuned model with DPO to mitigate potential harmful behaviors like hallucinations and promote responsible AI development. By open-sourcing its models, datasets, and fine-tuning code, xGen-MM aims to empower the research community and foster advancements in the field of LMMs, making these powerful tools more accessible and encouraging further exploration of their capabilities.
</details>

### **InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning**

InstructBLIP enhances the BLIP-2 framework by introducing instruction tuning to its Query Transformer (Q-Former), enabling the model to extract instruction-aware visual features and achieve state-of-the-art zero-shot performance across diverse vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2305.06500v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.06500v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hysts/InstructBLIP)  
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/5839e3a6-6fb8-469c-b84e-d60a851c1642" alt="Architecture diagram for InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**InstructBLIP**: represents an advanced step in the development of vision-language models through instruction tuning, building on the capabilities of the pre-trained BLIP-2 models. It integrates an image encoder, a large language model (LLM), and **a Query Transformer (Q-Former)**, which is specifically fine-tuned to bridge the visual and linguistic components while keeping the image encoder and LLM static. This architecture enables the extraction of instruction-aware visual features, enhancing the model's responsiveness to varied instructional contexts. Training InstructBLIP involves a careful selection of 26 datasets across 11 task categories, transformed into an instruction tuning format to foster the model's adaptability across a broad spectrum of vision-language tasks. The model employs a balanced sampling strategy and standard language modeling loss, augmented with OCR tokens for datasets involving scene texts, to fine-tune its instruction following capabilities. The unique approach of instruction-aware visual feature extraction through the Q-Former allows the model to tailor feature extraction to the specific requirements of the instruction, significantly improving performance across both seen and unseen tasks. Implementation details reveal the flexibility of InstructBLIP's architecture, which is easily adaptable to incorporate various LLMs, thanks to the modular design of the BLIP-2 framework. The model showcases state-of-the-art zero-shot performance across a wide range of vision-language tasks, outperforming previous models like BLIP-2 and Flamingo in zero-shot evaluations and achieving notable results when fine-tuned on specific downstream tasks. InstructBLIP's open-source availability and its performance across different benchmarks highlight its potential as a general-purpose vision-language model.
</details> 

### **KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models**

KOSMOS-1, a multimodal large language model, leverages a Transformer-based architecture enhanced with MAGNETO and XPOS to seamlessly process text and various modalities, aligning perception with language models through training on diverse web-scale multimodal corpora for enhanced zero-shot and few-shot learning capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2302.14045-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2302.14045) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/unilm)  
Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/33fd99a9-e89a-4905-8917-f03452fd5e6a" alt="Architecture diagram for KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**KOSMOS-1**: A transformative multimodal large language model, meticulously designed to harmonize the perception of general modalities with linguistic models, facilitating zero-shot learning, few-shot learning, and auto-regressive output generation. At its core, KOSMOS-1 employs a Transformer-based causal language model architecture, adept at processing both textual and various other modalities. This innovative approach is bolstered by key architectural components, including a Transformer-based decoder for input sequence handling, embedding modules for vector encoding of text and modalities, and the integration of **MAGNETO and XPOS** for architectural enhancements. These elements collectively enable the model to adeptly navigate and process multimodal information. The training regimen of KOSMOS-1 is distinguished by its comprehensive utilization of web-scale multimodal corpora, which encompasses monomodal data, cross-modal paired data, and interleaved multimodal data, emphasizing the next-token prediction tasks to optimize the log-likelihood of tokens. This methodology ensures a robust foundation for the model, enhancing its ability to understand and generate content across various modalities. Furthermore, the alignment techniques employed are particularly noteworthy; by leveraging interleaved image-text data, KOSMOS-1 aligns the perceptual capabilities of general modalities with language models in an unprecedented manner, thereby enriching the model's understanding and interpretative capacities. KOSMOS-1's training datasets, including The Pile, Common Crawl, English LAION-2B, LAION-400M, COYO-700M, and Conceptual Captions, are meticulously selected to serve dual purposes: fostering representation learning and language tasks through text corpora, and aligning perception with language models via image-caption pairs and interleaved data. This strategic selection of datasets not only bolsters the model's linguistic competencies but also significantly enhances its few-shot abilities, marking a significant milestone in the integration of perception and language models.
</details> 

### **KOSMOS-2: Grounding Multimodal Large Language Models to the World**

KOSMOS-2, extending the KOSMOS-1 architecture, incorporates grounded image-text pairs using discrete location tokens linked to text spans, effectively anchoring text to specific image regions, thereby enhancing multimodal understanding and reference accuracy.

[![arXiv](https://img.shields.io/badge/arXiv-2306.14824-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2306.14824) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/unilm/tree/master/kosmos-2) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ydshieh/Kosmos-2)  
Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/17420c9c-759d-4690-bfc8-e8d7792111e7" alt="Architecture diagram for KOSMOS-2: Grounding Multimodal Large Language Models to the World" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**KOSMOS-2**: Built upon the foundational architecture of KOSMOS-1, it retains the Transformer-based causal language model architecture and training objectives, while introducing a significant innovation by incorporating grounded image-text pairs into its training regimen. This addition seeks to bridge the gap between visual and textual information, enabling a more cohesive understanding of multimodal content. The model differentiates itself by training on a web-scale dataset of grounded image-text pairs, known as GRIT, which includes continuous coordinates of bounding boxes translated into discrete location tokens. These tokens are intricately linked with text spans, creating a unified input representation that seamlessly integrates visual and textual elements. The training of KOSMOS-2 is extensive and multifaceted, utilizing grounded image-text pairs, monomodal text corpora, image-caption pairs, and interleaved image-text data to foster a robust learning environment. The model's training leverages a large batch size and employs the AdamW optimizer, running on 256 V100 GPUs. This process is augmented by instruction tuning with both vision-language and language-only instruction datasets, aiming to refine the model's understanding and processing capabilities across different modalities. The grounding technique is a pivotal aspect of KOSMOS-2, where **continuous coordinates of bounding boxes** are converted into **discrete location tokens**. These tokens are then linked with corresponding text spans, anchoring the textual output to specific visual inputs, enhancing the model's ability to refer to and describe particular image regions or objects with precision. KOSMOS-2's alignment techniques and fusion methods play a critical role in its ability to understand and refer to specific parts of an image directly, employing a unified input representation that combines image embeddings with grounded text and location tokens. This approach not only improves the model's referential accuracy but also its overall multimodal comprehension. The model is trained using a variety of datasets, including the specially created GRIT dataset for grounding capabilities, along with monomodal text corpora, image-caption pairs, and interleaved image-text data to bolster its language understanding, multimodal perception, and in-context learning abilities. Through these innovations, KOSMOS-2 represents a significant advancement in grounding multimodal large language models, offering enhanced capabilities in linking textual and visual information cohesively.
</details> 

### **ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models**

ConvLLaVA addresses the limitations of Vision Transformers (ViTs) in high-resolution Large Multimodal Models (LMMs) by replacing them with a hierarchical backbone, ConvNeXt, as the visual encoder. This architectural shift aims to reduce the computational burden caused by excessive visual tokens and quadratic complexity often associated with ViTs, especially when dealing with high-resolution images.

[![arXiv](https://img.shields.io/badge/arXiv-2405.15738-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.15738) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alibaba/conv-llava) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2405.15738)  
Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, Bo Zheng  

<p align="center">
<img src="https://github.com/user-attachments/assets/ad7e129a-f958-4b30-8327-7df509994bea" alt="Architecture diagram for ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

ConvLLaVA leverages the inherent information compression capabilities of ConvNeXt, a hierarchical convolutional neural network. ConvLLaVA, unlike traditional LMMs that rely on ViTs, employs a **five-stage ConvNeXt architecture** as its visual encoder. This encoder progressively compresses visual information across its stages, significantly reducing the number of visual tokens generated compared to ViT. The architecture mirrors other popular general LMMs like LLaVA, Qwen-VL, and VILA, consisting of a vision encoder (ConvNeXt), a large language model (LLM - Vicuna in this case), and a vision-language projector (a two-layer MLP). The ConvNeXt encoder processes the input image and generates latent visual embeddings. These embeddings are then projected into the embedding space of the LLM by the vision-language projector. Finally, the projected visual embeddings are concatenated with the text embeddings generated by the LLM's tokenizer, and this combined input is fed into the LLM. The entire model is trained using a language modeling loss. To further enhance ConvLLaVA's performance, the authors introduce two key optimizations: firstly, they update the pretrained ConvNeXt weights instead of freezing them, allowing the model to adapt to high-resolution inputs and improve the quality of visual representations. Secondly, they introduce an additional ConvNeXt stage, effectively creating a five-stage architecture (ConvNeXt†) that further compresses visual information, enabling the model to handle even higher resolutions (up to 1536x1536) while generating a manageable number of visual tokens (576). This hierarchical compression approach, combined with the linear spatial complexity of ConvNeXt, significantly reduces the computational burden on the LLM compared to ViT-based models, making ConvLLaVA a more efficient and scalable solution for high-resolution multimodal tasks.
</details>

### **Parrot: Multilingual Visual Instruction Tuning**

Parrot tackles the issue of "multilingual erosion" in Multimodal Large Language Models (MLLMs), where models trained primarily on English-centric data struggle to understand and respond in other languages. It achieves this by using textual guidance to align visual tokens with language-specific embeddings, effectively enhancing the model's multilingual capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2406.02539-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.02539) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/AIDC-AI/Parrot)  
Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, Han-Jia Ye  

<p align="center">
<img src="https://github.com/user-attachments/assets/467964a0-4ccc-4cec-802a-c93b310d3118" alt="Architecture diagram for Parrot: Multilingual Visual Instruction Tuning" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Parrot builds upon the LLaVA framework, utilizing a pre-trained CLIP ViT-L/14 as the vision encoder and Qwen1.5-Chat as the LLM. The architecture consists of three main components: a vision encoder, a large language model (LLM), and a multilingual **Mixture-of-Experts (MoE)** module. The vision encoder processes the input image and generates visual features, which are then projected into the embedding space of the LLM using a learned projector. To address the multilingual challenge, Parrot introduces a novel textual guidance mechanism. It first calculates cross-attention between the class token of the visual features and the text embeddings derived from the input prompt. This cross-attention output is then fed into the MoE module's router, which predicts the probability of activating each language expert. Each expert is a specialized MLP trained to transform the English-biased visual embeddings into language-specific representations. The router selects the most relevant experts based on the input language, and their outputs are combined to generate the final language-specific visual embeddings. These embeddings are then combined with the original visual embeddings using a weighted sum, ensuring that the model retains its ability to process visual information effectively across different languages. This entire process allows Parrot to align visual tokens with textual embeddings at the language level, effectively mitigating multilingual erosion and enhancing the model's ability to understand and respond in multiple languages.
</details>

### **OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding**

OMG-LLaVA presents a novel framework that unifies image-level, object-level, and pixel-level reasoning and understanding within a single Multimodal Large Language Model (MLLM). It leverages the power of a frozen universal segmentation model (OMG-Seg) for visual encoding and a Large Language Model (LLM) for text understanding and response generation, enabling a wide range of multimodal tasks within a single, elegant architecture.

[![arXiv](https://img.shields.io/badge/arXiv-2406.19389-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2406.19389) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/lxtGH/OMG-Seg) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2406.19389)  
Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, Shuicheng Yan  

<p align="center">
<img src="https://github.com/user-attachments/assets/c2830cc5-ab00-4c48-898e-a077cdc7b947" alt="Architecture diagram for OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

OMG-LLaVA consists of two main components: a frozen universal perception module (based on OMG-Seg) and a Large Language Model (LLM). The universal perception module is responsible for encoding the input image and visual prompts into three types of visual tokens: pixel-centric, object-centric, and object-centric derived from visual prompts. The pixel-centric tokens are generated by a **ConvNeXt-L based CLIP image encoder**, capturing dense image features. The object-centric tokens are generated by the OMG decoder, which takes learnable object queries and visual prompt queries as input and attends to the image features to extract object-level information. This decoder can handle point, box, and mask prompts by applying constraints on the attention masks. To bridge the gap between the frozen perception module and the LLM, a novel "perception prior embedding" strategy is introduced. This strategy fuses the image features with the object queries from the OMG decoder using a mask score derived from the segmentation masks and confidence scores. The resulting weighted object queries are then added to the image features to generate the pixel-centric visual tokens, providing the LLM with rich object-level information. The object-centric visual tokens are directly taken from the foreground object queries of the OMG decoder. Both types of visual tokens, along with the text instruction tokens, are fed into the LLM, which is responsible for understanding the user's intent and generating the appropriate response. The LLM outputs text responses and object-centric visual tokens, which are then decoded by the frozen OMG decoder to produce segmentation masks. This unified architecture allows OMG-LLaVA to perform a wide range of tasks, including image captioning, visual question answering, referring segmentation, reasoning segmentation, grounded conversation generation, and region captioning, all within a single model.
</details>

### **EVLM: An Efficient Vision-Language Model for Visual Understanding**

EVLM is an efficient multimodal language model designed to minimize computational costs while maximizing the model's ability to perceive visual signals comprehensively. It addresses the challenges of handling long sequences of visual signals, particularly in video data, by employing a cross-attention mechanism and hierarchical ViT features, achieving competitive performance in tasks like image and video captioning.

[![arXiv](https://img.shields.io/badge/arXiv-2407.14177-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.14177) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.14177)  
Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, Changyi Liu, Dewen Fan, Huihui Xiao, Jiahong Wu, Fan Yang, Size Li, Di Zhang  

<p align="center">
<img src="https://github.com/user-attachments/assets/87563a37-e65e-44d4-a0e1-aea452ae313c" alt="Architecture diagram for EVLM: An Efficient Vision-Language Model for Visual Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

EVLM is built upon the Flamingo architecture, incorporating a visual encoder, a large language model, and a Gated Cross-Attention Layer. To enhance visual perception, EVLM utilizes the 4.4B EVA2-CLIP-E-Plus model as the visual encoder, extracting hierarchical visual features by uniformly sampling 8 feature sequences from the last 40 layers of the transformer. These features are then sequentially fed into different Gated Cross-Attention layers of the Flamingo model. Unlike Flamingo, which uses a single media token image, EVLM replaces it with a set of 16 learnable tokens, aiming to capture visual features similar to Q-former. The attention mechanism is designed to allow each set of learnable tokens to interact only with the corresponding image, while text sequences interact only with the previous image in the multimodal sequence. This approach ensures efficient interaction between visual and textual information. For the language model, EVLM employs the Qwen-14B-Chat 1.0, chosen for its strong performance in content understanding and logical reasoning. A gated cross-attention layer is inserted before every transformer layer of the language model to condition it on visual inputs. To further enhance model effectiveness and scale trainable parameters, a Mixture of Experts (MoE) mechanism is applied to the Cross Attention layer. This involves replicating and segmenting the FFN of the base model into multiple fine-grained experts, with a routing layer selecting the appropriate set of experts for each token. The model undergoes a three-stage training process: multi-modal pre-training, multi-task continual pre-training, and multi-modal instruction fine-tuning. Pre-training focuses on cross-modal alignment and modeling intrinsic relationships within multimodal data, using a large-scale dataset of bilingual image-text captions and web-type multimodal data. Continual pre-training further enhances the model's visual question-answering ability, while instruction fine-tuning activates its instruction-following capabilities using a diverse range of high-quality instruction tuning data.
</details>

### **SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models**

SlowFast-LLaVA (SF-LLaVA) is a training-free video large language model that effectively captures both detailed spatial semantics and long-range temporal context in videos without requiring any additional fine-tuning on video data. It achieves this by leveraging a two-stream SlowFast design inspired by action recognition models, allowing it to process a larger number of frames and outperform existing training-free methods on various video benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2407.15841-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.15841) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.15841)  
Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, Afshin Dehghan  

<p align="center">
<img src="https://github.com/user-attachments/assets/6e1e2f43-86a7-42e3-998a-24bbd8f1c741" alt="Architecture diagram for SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

SF-LLaVA builds upon the LLaVA-NeXT framework and utilizes a two-stream approach, similar to SlowFast networks in action recognition, to process video inputs. The model first uniformly samples N frames from the input video. These frames are then processed independently by a visual encoder, such as CLIP-L, followed by a visual-language adapter for feature alignment. The resulting frame features are then fed into two separate pathways: Slow and Fast. **The Slow pathway** focuses on capturing detailed spatial semantics by processing a smaller number of frames (Nslow) at a higher spatial resolution (e.g., 8 frames with 24x24 tokens). It applies spatial pooling with a small stride (e.g., 1x2) to aggregate features and reduce the number of tokens. **The Fast pathway** focuses on capturing temporal context and motion cues by processing all N frames (Nfast = N) at a lower spatial resolution (e.g., 64 frames with 4x4 tokens). It applies aggressive spatial pooling to each frame to prioritize temporal information. The features from both pathways are then flattened and concatenated, forming a comprehensive video representation that balances spatial details and temporal context. This aggregated feature vector, along with the text prompt and question, is then fed into the LLM (LLaVA-NeXT) to generate the final answer. This training-free approach eliminates the need for expensive fine-tuning on video datasets, making SF-LLaVA highly efficient and adaptable to various video scenarios. The authors demonstrate the effectiveness of SF-LLaVA on three different video question-answering tasks (Open-Ended VideoQA, Multiple Choice VideoQA, and Text Generation) across eight benchmarks, showcasing its superior performance compared to existing training-free methods and even surpassing some state-of-the-art supervised fine-tuned video LLMs.
</details>

### **INF-LLaVA: High-Resolution Image Perception for Multimodal Large Language Models**

INF-LLaVA is a novel Multimodal Large Language Model (MLLM) designed to effectively process high-resolution images. It addresses the limitations of existing cropping-based and dual-encoder methods by introducing two innovative modules: Dual-perspective Cropping Module (DCM) and Dual-perspective Enhancement Module (DEM). DCM segments high-resolution images into sub-images from both local and global perspectives, preserving detailed and contextual information. DEM facilitates efficient interaction between local and global features, enhancing the model's understanding of complex visual relationships. Extensive evaluations demonstrate INF-LLaVA's superior performance on various benchmarks, establishing a new state-of-the-art in vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2407.16198-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.16198) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/WeihuangLin/INF-LLaVA) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.16198)  
Yiwei Ma, Zhibin Wang, Xiaoshuai Sun, Weihuang Lin, Qiang Zhou, Jiayi Ji, Rongrong Ji  

<p align="center">
<img src="https://github.com/user-attachments/assets/641027c4-a5eb-42e8-8486-b58f3508c553" alt="Architecture diagram for INF-LLaVA: High-Resolution Image Perception for Multimodal Large Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

INF-LLaVA pushes the boundaries of Multimodal Large Language Models (MLLMs) by tackling the critical challenge of high-resolution image perception. It aims to leverage the richness of detail present in high-resolution images without succumbing to the computational limitations imposed by traditional MLLM architectures. INF-LLaVA achieves this through a sophisticated approach that combines innovative cropping and feature enhancement techniques, resulting in a model capable of simultaneously capturing fine-grained local details and comprehensive global context. At the core of INF-LLaVA lies the Dual-perspective Cropping Module (DCM), a strategic cropping strategy that surpasses conventional approaches by integrating both local and global perspectives. This dual-perspective approach ensures that each extracted sub-image retains not only the intricate details essential for accurate analysis but also the broader contextual information crucial for understanding the relationships between objects. While local-perspective cropping preserves continuous visual information at high resolution, capturing the essence of individual objects and regions, global-perspective cropping leverages a unique interleaving technique to preserve the overall spatial relationships between objects within the high-resolution image. This balanced combination ensures that the model can perceive both the "trees" and the "forest," enabling a holistic understanding of the visual scene. To further enhance the model's understanding, INF-LLaVA introduces the Dual-perspective Enhancement Module (DEM). This module facilitates efficient and effective interaction between the local and global features extracted by the vision encoder, enriching the representation with multi-scale information. Instead of relying on computationally expensive cross-attention directly on high-resolution features, DEM employs a more resource-efficient strategy. It leverages 2D positional priors to concatenate global-perspective sub-image features back into the original image's shape, effectively recreating a high-resolution representation of the global context. These recombined features are then re-cropped from a local perspective, and cross-attention is performed between corresponding local and global sub-images to enhance global features with fine-grained local details. A symmetrical process enhances local features with global context. This meticulously designed interaction between local and global features ensures that the resulting representation is not only rich in detail but also cognizant of the broader context. The dual-enhanced features are then projected into a format compatible with the LLM through a linear connector. The LLM then processes the combined visual and textual information to generate a coherent and contextually relevant response. Through extensive evaluations on a diverse set of benchmarks, including ScienceQA-img, OKVQA, SEEDBench, MMBench, AI2D, LLaVA-Bench-in-the-wild, and MMMU, INF-LLaVA demonstrates its superior performance over existing MLLMs. Its ability to effectively handle high-resolution images while maintaining computational efficiency establishes a new state-of-the-art in the field. The open-source release of INF-LLaVA, along with its pretrained model and code, paves the way for further research and exploration of high-resolution image perception in multimodal large language models, pushing the boundaries of multimodal understanding and enabling the development of more powerful and versatile AI systems.
</details>


### **VILA²: VILA Augmented VILA**

VILA² (VILA-augmented-VILA) introduces a novel approach to address the limitations of data quantity and quality in training Visual Language Models (VLMs). Instead of relying on costly human annotation or distillation from proprietary models, VILA² leverages the VLM itself to iteratively refine and augment its pretraining data, leading to significant performance improvements and achieving state-of-the-art results on the MMMU leaderboard among open-sourced models.

[![arXiv](https://img.shields.io/badge/arXiv-2407.17453-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.17453) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2407.17453)  
Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jang Hyun Cho, Marco Pavone, Song Han, Hongxu Yin  

<p align="center">
<img src="https://github.com/user-attachments/assets/b7602734-1163-49aa-bf78-27ae42a520bd" alt="Architecture diagram for VILA&#178;: VILA Augmented VILA" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

VILA² employs a two-step iterative process: self-augmenting and specialist-augmenting. The self-augmenting loop focuses on enhancing the general knowledge of the VLM by using the model itself to re-caption its pretraining data. This process starts with an initial VLM (VILA0) trained on a dataset with typically short and brief captions, like COYO. VILA0 is then used to generate longer and more detailed captions for the same images, creating a synthetic dataset. This augmented dataset, combined with the original data, is used to train the next iteration of the VLM (VILA1). This loop can be repeated multiple times, with each iteration improving the caption quality and subsequently the VLM's performance. However, this self-augmentation process eventually reaches saturation. To overcome this limitation, VILA² introduces the **specialist-augmenting loo**p. This involves fine-tuning the self-augmented VLM on specific downstream tasks, creating specialist VLMs with expertise in areas like spatial awareness, OCR, and grounding. These specialists are then used to re-caption the pretraining data, focusing on their specific domain knowledge. The self-augmented VLM is then retrained on this specialist-recaptioned data, further boosting its performance. This approach leverages the synergy between the vast amount of data in pretraining and the specialized knowledge acquired during fine-tuning. The architecture of VILA² follows the standard auto-regressive VLM design, consisting of a large language model (LLM), a visual encoder, and an image-text projector. The authors experiment with different LLMs (Llama2-7B, Llama3-8B-Instruct, and Yi-34B) and visual encoders (SigLIP and InternViT-6B). They also introduce a 4x downsampling of visual tokens to reduce computational cost. The training process follows the typical three-stage paradigm: projector initialization, vision-language pre-training, and visual instruction-tuning. VILA² demonstrates significant performance improvements over previous state-of-the-art methods on various benchmarks, including general VQA, text-oriented VQA, general multimodal benchmarks, and image captioning. This highlights the effectiveness of the proposed self- and specialist-augmentation techniques in enhancing VLM training and achieving state-of-the-art results.
</details>

### **MiniCPM-V: A GPT-4V Level MLLM on Your Phone**

MiniCPM-V is a series of efficient Multimodal Large Language Models (MLLMs) designed for deployment on end-side devices like mobile phones and personal computers. The latest iteration, MiniCPM-Llama3-V 2.5, achieves performance comparable to GPT-4V, Gemini Pro, and Claude 3 while being significantly smaller and more efficient, demonstrating the feasibility of deploying powerful MLLMs on resource-constrained devices.

[![arXiv](https://img.shields.io/badge/arXiv-2408.01800-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.01800) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenBMB/MiniCPM-V) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/openbmb/MiniCPM-V-2_6)  
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, Maosong Sun  

<p align="center">
<img src="https://github.com/user-attachments/assets/d943871a-ca05-46d6-9572-7fe02dda1495" alt="Architecture diagram for MiniCPM-V: A GPT-4V Level MLLM on Your Phone" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

MiniCPM-V focuses on achieving a balance between performance and efficiency, crucial for real-world applications on end-side devices. The model architecture consists of three key modules: a visual encoder, a compression layer, and an LLM. For the visual encoder, MiniCPM-V utilizes SigLIP SoViT-400m/14, chosen for its efficiency and effectiveness. To handle high-resolution images with varying aspect ratios, the model employs an adaptive visual encoding approach. This involves dividing the input image into slices that better match the ViT's pre-training settings in terms of resolution and aspect ratio. A score function is used to select the optimal partition of slices, ensuring a good match with the ViT's pre-training. Each slice is then resized proportionally and interpolated to fit the ViT's input size. After visual encoding, each slice is represented by 1024 tokens, resulting in a large number of tokens for multiple slices. To address this, a token compression module is employed, using one-layer cross-attention with a moderate number of queries to compress the visual tokens of each slice into 64 or 96 tokens. This significantly reduces the computational cost and memory footprint, making the model suitable for end-side deployment. A spatial schema is also introduced to indicate the position of each slice relative to the whole image, further enhancing the model's understanding of spatial relationships. The compressed visual tokens, along with the text input, are then fed into the LLM, which is based on MiniCPM 2B for earlier versions and Llama3-Instruct 8B for MiniCPM-Llama3-V 2.5. The training process consists of three phases: pre-training, supervised fine-tuning, and RLAIF-V (Reinforcement Learning from AI Feedback for Vision). Pre-training aims to align the visual modules with the LLM's input space and learn foundational multimodal knowledge. It involves three stages: warming up the compression layer, extending the input resolution of the visual encoder, and training the visual modules with the adaptive visual encoding strategy. Supervised fine-tuning further enhances the model's knowledge and interaction capabilities using high-quality visual question answering datasets. The SFT data is categorized into two parts: one focusing on basic recognition capabilities and the other on generating detailed responses and following instructions. Finally, RLAIF-V is employed to mitigate the hallucination problem common in MLLMs. This involves generating multiple responses for an instruction, evaluating their correctness using a divide-and-conquer strategy, and then optimizing the model using Direct Preference Optimization (DPO) on a preference dataset. MiniCPM-V demonstrates impressive performance on various benchmarks, including general multimodal benchmarks, OCR benchmarks, and multilingual multimodal interaction, while being efficient enough for deployment on mobile phones. This highlights the potential of pushing the boundaries of end-side MLLMs and bringing powerful AI capabilities to user devices.

**MiniCPM-V 4.5**, released in September 2025, keeps the family's efficient visual pathway and compact language backbone while introducing a unified 3D-Resampler for highly compressed image and video encoding, a unified document-knowledge and text-recognition recipe, and hybrid reinforcement learning for short and long reasoning modes. **MiniCPM-V 4.6**, released in May 2026, returns to an approximately 1B edge footprint with SigLIP2-400M, Qwen3.5-0.8B, LLaVA-UHD v4, and switchable 4x/16x visual-token compression. Both are lineage updates rather than separate top-level architectures. [4.5 paper](https://arxiv.org/abs/2509.18154) · [4.6 model](https://huggingface.co/openbmb/MiniCPM-V-4.6) · [Repository](https://github.com/OpenBMB/MiniCPM-V)
</details>

### **MiniCPM-o-2.6: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming**

MiniCPM-o-2.6 is a powerful 8B parameter multimodal large language model (MLLM) that excels in vision, speech, and multimodal live streaming, achieving performance comparable to GPT-4o in several benchmarks, while maintaining high efficiency for deployment on edge devices.

[![arXiv](https://img.shields.io/badge/Blog-MiniCPM%20Team%20Blog-b31b1b.svg?style=flat-square)](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OpenBMB/MiniCPM-o)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/openbmb/MiniCPM-o-2_6)  
OpenBMB

<p align="center">
<img src="https://github.com/user-attachments/assets/cb066a40-8da7-4775-b002-7c054697f1ec" width=600 alt="Architecture diagram for MiniCPM-o-2.6: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

MiniCPM-o-2.6 employs an end-to-end omni-modal architecture.  It integrates several pre-trained components: **Vision Encoder:** SigLip-400M **Audio Encoder:** Whisper-medium-300M **Text-to-Speech (TTS):** ChatTTS-200M **Large Language Model (LLM):** Qwen2.5-7B. These components are connected and trained end-to-end.  A key innovation is the "Omni-modal Live Streaming Mechanism." This involves: **Online Modality Encoders/Decoders:**  The offline encoders and decoders are transformed into online versions to handle streaming inputs and outputs. **Time-Division Multiplexing (TDM):** A TDM mechanism within the LLM backbone processes omni-modal streams. It divides parallel streams (video, audio) into sequential information within short time slices.  **Configurable Speech Modeling:** A multimodal system prompt (including text and audio prompts) allows for flexible voice configuration during inference, enabling voice cloning and description-based voice creation.

**MiniCPM-o 4.5**, released in April 2026, evolves this family toward real-time full-duplex omni-modal interaction. It retains streaming vision, speech, and text while improving simultaneous perception and generation; the update belongs here because it extends the same time-multiplexed, end-to-end omni-modal lineage. [Paper](https://arxiv.org/abs/2604.27393) · [Repository](https://github.com/OpenBMB/MiniCPM-V)
</details>

### **LLaVA-OneVision: Easy Visual Task Transfer**

LLaVA-OneVision is a family of open large multimodal models (LMMs) designed to excel in various computer vision scenarios, including single-image, multi-image, and video understanding. It pushes the performance boundaries of open LMMs by consolidating insights from the LLaVA-NeXT blog series, focusing on data, models, and visual representations. Notably, LLaVA-OneVision demonstrates strong transfer learning capabilities, enabling it to excel in video understanding tasks by leveraging knowledge learned from image data.

[![arXiv](https://img.shields.io/badge/arXiv-2408.03326-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.03326) [![Website](https://img.shields.io/badge/🌐-Website-blue)](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/papers/2408.03326)  
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li  

<p align="center">
<img src="https://github.com/user-attachments/assets/abe36db3-571d-4068-b532-7512d4a5fcc5" alt="Architecture diagram for LLaVA-OneVision: Easy Visual Task Transfer" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

LLaVA-OneVision inherits the minimalist design of the LLaVA series, aiming to effectively leverage pre-trained capabilities of both the LLM and the visual model while facilitating strong scaling. The architecture consists of three key components: a large language model (LLM), a vision encoder, and a projector. The authors choose Qwen-2 as the LLM due to its strong language capabilities and various model sizes available. For the vision encoder, they opt for SigLIP, which has shown to yield higher LMM performance among open vision encoders. A 2-layer MLP is used as the projector to map image features into the word embedding space, creating a sequence of visual tokens. The model utilizes a flexible visual representation strategy called Higher AnyRes, which builds upon the original AnyRes strategy introduced in LLaVA-NeXT. This strategy involves dividing the input image into crops, each with a resolution suitable for the vision encoder, and then applying bilinear interpolation to reduce the number of tokens per crop if needed. This allows the model to handle high-resolution images and videos efficiently while preserving important visual details. The specific configuration of **Higher AnyRes** is adapted for different scenarios: single-image, multi-image, and video. For single-image data, a large maximum spatial configuration is used to maintain the original image resolution and a large number of visual tokens are allocated to effectively represent the visual signal. For multi-image data, only the base image resolution is considered, eliminating the need for multi-crop and saving computational resources. For video data, each frame is resized to the base image resolution and bilinear interpolation is used to reduce the number of tokens per frame, allowing for the processing of a larger number of frames. The training process follows a three-stage curriculum learning approach: language-image alignment, high-quality knowledge learning, and visual instruction tuning. The first stage focuses on aligning visual features with the LLM's embedding space using the LLaVA align dataset. The second stage refines and enhances the model's knowledge base using high-quality data from three major categories: re-captioned detailed description data, document/OCR data, and Chinese and language data. The final stage involves visual instruction tuning, where the model is trained on a diverse set of visual tasks with preferred responses. This stage is further divided into two phases: single-image training and OneVision training. Single-image training focuses on single-image scenarios, while OneVision training expands the model's capabilities to multi-image and video scenarios, enabling task transfer and emerging capabilities. LLaVA-OneVision demonstrates state-of-the-art performance on various benchmarks, including single-image, multi-image, and video tasks, showcasing its effectiveness and versatility in handling diverse visual scenarios.

Released by arXiv v1 on September 28, 2025, **LLaVA-OneVision-1.5** preserves the established vision-encoder-projector-language-model design but makes the complete construction pipeline reproducible: an 85M-sample concept-balanced mid-training set, a 26M-sample instruction set, 64B compressed multimodal tokens, and offline parallel data packing reduce reported training cost to approximately $16,000. It is a major open-training update, but architecturally remains part of LLaVA-OneVision. [Paper](https://arxiv.org/abs/2509.23661) · [Repository](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)
</details>

### **VITA: Towards Open-Source Interactive Omni Multimodal LLM**

VITA is the first open-source Multimodal Large Language Model (MLLM) capable of simultaneously processing and analyzing video, image, text, and audio modalities while offering an advanced multimodal interactive experience. It addresses the limitations of existing open-source models, which often excel in either understanding or interaction but rarely both, by integrating architectural innovations with advanced training and development strategies.

[![arXiv](https://img.shields.io/badge/arXiv-2408.05211-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.05211) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/VITA-MLLM/VITA) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/VITA-MLLM)  
Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, Xing Sun

<p align="center">
<img src="https://github.com/user-attachments/assets/94e2b781-0c86-47df-ac18-76ebc71bb349" alt="Architecture diagram for VITA: Towards Open-Source Interactive Omni Multimodal LLM" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

VITA starts with the Mixtral 8x7B model as its language foundation, chosen for its strong performance and sparse mixture of experts (SMoE) architecture. To enhance its Chinese language capabilities, the vocabulary is expanded with Chinese terms, and the model undergoes bilingual instruction tuning using a high-quality bilingual text corpus. This ensures proficiency in both Chinese and English. For visual modality, VITA employs InternViT-300M-448px as the visual encoder, processing images at 448x448 resolution and generating 256 tokens after passing through a two-layer MLP visual connector. High-resolution images are handled using a dynamic patching strategy, while videos are treated as special cases of images, with frame sampling based on video length. For audio modality, a Mel Filter Bank block is used to process the input audio, followed by 4xCNN downsampling layers and a 24-layer transformer, resulting in 25 tokens for every 2 seconds of audio. A two-layer MLP serves as the audio-text modality connector. The training pipeline consists of three stages: LLM instruction tuning, multimodal alignment, and multimodal instruction tuning. LLM instruction tuning focuses on enhancing the base LLM's bilingual capabilities. Multimodal alignment aims to bridge the representation gap between text and other modalities by training individual encoders and connectors for each modality. This involves collecting and curating a large-scale, high-quality multimodal dataset, including image descriptions, general image QA, OCR and diagram data, general video descriptions, general video QA, and pure text data. Multimodal instruction tuning further refines the model's ability to follow instructions and understand different modalities. A specially designed state token is introduced to distinguish the type of input query (effective audio, noisy audio, or text), enabling non-awakening interaction during inference. To achieve natural multimodal human-computer interaction, VITA introduces two key innovations: non-awakening interaction and audio interrupt interaction. These are implemented using a duplex pipeline during deployment. Two VITA models run concurrently: one for generating responses to user queries (Generation model) and the other for monitoring environmental audio (Monitoring model). The Monitoring model uses SileroVAD for voice activity detection and filters out noisy audio based on the state token. If an effective audio query is detected, the Monitoring model interrupts the Generation model, consolidates the historical context, and responds to the latest query. The two models then swap identities, ensuring continuous monitoring and seamless interaction.VITA demonstrates strong performance on various unimodal and multimodal benchmarks, showcasing its robust foundational capabilities in multilingual, vision, and audio understanding. While still lagging behind closed-source counterparts in certain areas, VITA represents a significant step towards open-source interactive omni-modal LLMs, paving the way for future research and development in this field.
</details>

### **EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders**

EAGLE is a family of open-source Multimodal Large Language Models (MLLMs) that leverage a mixture of vision encoders to achieve state-of-the-art performance on various benchmarks, particularly in tasks involving OCR and document understanding. The study focuses on systematically exploring the design space of MLLMs with multiple vision encoders, aiming to identify optimal design choices and improve MLLM perception.

[![arXiv](https://img.shields.io/badge/arXiv-2408.15998-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2408.15998) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/NVlabs/EAGLE) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/NVEagle/Eagle-X5-13B-Chat)  
Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, Guilin Liu  

<p align="center">
<img src="https://github.com/user-attachments/assets/4e057a78-3fad-4a04-9a05-0f5361a8255b" alt="Architecture diagram for EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

EAGLE builds upon the LLaVA architecture, consisting of a large language model, a vision encoder, and a projection layer. The core innovation lies in integrating multiple vision experts, each pre-trained on different tasks and resolutions, to enhance the model's ability to perceive and comprehend diverse visual information. The study explores various aspects of this design space, including high-resolution adaptation, fusion paradigms, and optimal encoder combinations. It introduces a Pre-Alignment training stage to address representational inconsistencies between vision-focused encoders and language tokens. The training process consists of three progressive stages: vision-language pre-alignment, joint-projector training, and supervised fine-tuning. EAGLE achieves state-of-the-art performance on various benchmarks, demonstrating significant advantages in OCR and document understanding tasks. The study highlights the importance of systematic design space exploration and the effectiveness of combining multiple vision experts with a streamlined fusion strategy and a pre-alignment training stage for building high-performing MLLMs.
</details>

### **Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models**

Eagle 2 is a family of vision-language models (VLMs) developed with a data-centric approach, focusing on post-training data strategies to achieve state-of-the-art performance. The models build upon open-source components and prioritize data diversity and quality, using a three-stage training recipe and a tiled mixture of vision encoders (MoVE) architecture, achieving results that match or surpass those of larger, proprietary models.

[![arXiv](https://img.shields.io/badge/arXiv-2501.14818-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2501.14818)
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/NVlabs/EAGLE)
[![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/nvidia/Eagle2-9B)  
Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Yilin Zhao, Subhashree Radhakrishnan, Nadine Chang, Matthieu Le, De-An Huang, Ilia Karmanov, Lukas Voegtle, Jose M. Alvarez, Bryan Catanzaro, Jan Kautz, Andrew Tao, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Karan Sapra, Amala Deshmukh, Tuomas Rintamaki, Philipp Fischer, Timo Roman, Tong Lu, Guilin Liu, Zhiding Yu

<p align="center">
<img src="https://github.com/user-attachments/assets/e4280077-c80f-4cca-bd8f-3122a322520e" width="600" alt="Architecture diagram for Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

**Eagle 2** adopts a "diversity first, then quality" data strategy, beginning with a large, diverse pool of over 180 data sources, followed by rigorous filtering and selection. The architecture uses a tiled mixture of vision encoders (MoVE), specifically SigLIP and ConvNeXt-XXLarge, with image tiling to handle high resolutions.  Each image tile is encoded by channel-concatenated MoVE. The vision encoder outputs are concatenated and aligned with the LLM (Qwen2.5) via a simple MLP connector.  A three-stage training recipe is used: Stage 1 trains the connector to align modalities; Stage 1.5 trains the full model on a large, diverse dataset; and Stage 2 fine-tunes on a high-quality instruction-tuning dataset.  Crucially, *all* available visual instruction data is used in Stage 1.5, not just captioning/knowledge data.  Balanced data packing addresses limitations in existing open-source frameworks. The core contribution is the detailed data strategy.  This involves: (1) **Data Collection**: Building a highly diverse data pool (180+ sources) through both passive gathering (monitoring arXiv and Hugging Face) and proactive searching (addressing "bucket effect" via error analysis). (2) **Data Filtering**: Removing low-quality samples based on criteria like mismatched question-answer pairs, irrelevant image-question pairs, repeated text, and numeric formatting issues. (3) **Data Selection**: Choosing optimal subsets based on data source diversity, distribution, and K-means clustering on SSCD image embeddings to ensure balance across types (especially useful for chart data, etc.). (4) **Data Augmentation**: Mining information from input images through techniques like Chain-of-Thought (CoT) explanation generation, rule-based QA generation, and expanding short answers into longer ones. (5) **Data Formating:** remove unnecessary decorations. Training uses a three-stage approach:
**Stage 1:** Aligns language and image modalities by training the MLP connector.
**Stage 1.5:** Trains the *full* model using a large-scale, diverse dataset (21.6M samples). *All* available visual instruction data is used here, unlike common two-stage approaches, leading to substantial improvements.
**Stage 2:** Fine-tunes the full model on a carefully curated, high-quality visual instruction tuning dataset (4.6M samples). The model is trained with AdamW. Eagle 2 demonstrates strong performance across a wide range of multimodal benchmarks, matching or outperforming frontier open-source and some closed-source VLMs.
</details>

### **Florence-2: A Deep Dive into its Unified Architecture and Multi-Task Capabilities**

Florence-2 presents a significant advancement in vision foundation models, aiming to achieve a single, versatile representation capable of handling a wide spectrum of vision and vision-language tasks through a unified, prompt-based approach. Unlike previous models that often specialize in specific tasks, Florence-2 is designed to be a generalist, adept at performing tasks with simple text instructions, similar to how Large Language Models (LLMs) operate.

[![arXiv](https://img.shields.io/badge/arXiv-2311.06242-b31b1b.svg?style=flat-square)](https://arxiv.org/pdf/2311.06242) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/gokaygokay/Florence-2)  
Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, Lu Yuan  

<p align="center">
<img src="https://github.com/user-attachments/assets/f9c1f95b-ba6a-4a55-bf52-fa043b339d27" alt="Architecture diagram for Florence-2: A Deep Dive into its Unified Architecture and Multi-Task Capabilities" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

Florence-2 lies a sophisticated architecture comprised of two key components: an image encoder and a multi-modality encoder-decoder. The image encoder, powered by the powerful DaViT architecture, transforms the input image into a sequence of visual token embeddings, effectively capturing the visual information. These visual embeddings are then combined with text embeddings derived from task-specific prompts. This fusion of visual and linguistic information is processed by a standard transformer-based multi-modality encoder-decoder. This component acts as the brain of the model, meticulously analyzing the combined input and generating the desired output in textual form. This unified architecture, with a single set of parameters governing various tasks, eliminates the need for task-specific modifications, leading to a streamlined and efficient model. This design philosophy mirrors the trend in the NLP community, where models with consistent underlying structures are preferred for their versatility and ease of development. Florence-2's capabilities span a multitude of tasks, showcasing its remarkable adaptability. It excels at generating detailed image captions, capturing the essence of an image through rich textual descriptions. Its prowess extends to visual grounding, accurately pinpointing specific objects or regions within an image based on textual phrases. Florence-2 also demonstrates impressive performance in open-vocabulary object detection, identifying objects by their names, even if those objects were not part of its training data. This capability highlights the model's ability to generalize its knowledge and understand novel visual concepts. Furthermore, Florence-2 tackles dense region captioning, providing detailed descriptions for multiple regions within an image, and even performs optical character recognition (OCR), extracting text from images. This broad range of capabilities makes Florence-2 a powerful tool for numerous applications, pushing the boundaries of multimodal understanding in AI.
</details>

### **MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning**

MULTIINSTRUCT leverages the OFA model as its foundation, employing a Transformer-based sequence-to-sequence architecture and instruction tuning techniques on a diverse dataset, effectively aligning text and image tokens within a unified space for enhanced multi-modal zero-shot learning.

[![arXiv](https://img.shields.io/badge/arXiv-2212.10773-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2212.10773) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vt-nlp/multiinstruct)  
Zhiyang Xu, Ying Shen, Lifu Huang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/bedfc8b1-7aff-44af-b605-4470ad030bdf" alt="Architecture diagram for MULTIINSTRUCT: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MULTIINSTRUCT**: introduces a novel approach to enhance multi-modal zero-shot learning by leveraging instruction tuning, built upon the foundation of the **OFA (Omnipotent Fast Adapters)** as its core pre-trained multi-modal model. This model adopts a Transformer-based sequence-to-sequence architecture that efficiently encodes a mix of instructions, text, images, and bounding boxes within a unified token space. Such a design enables MULTIINSTRUCT to process and interpret a wide range of input types, including optional images, through a comprehensive encoder-decoder framework. The encoder component is dedicated to processing the diverse inputs and instructions, while the decoder is tasked with generating the corresponding outputs. At the heart of MULTIINSTRUCT's training methodology is the innovative use of the model-specific MULTIINSTRUCT dataset, alongside instruction tuning techniques that incorporate instances from multiple tasks. This approach involves a combination of random shuffling and sampling of instruction templates for batch training, significantly enriching the learning process. Furthermore, the model explores advanced transfer learning strategies through Mixed Instruction Tuning and Sequential Instruction Tuning, utilizing the NATURAL INSTRUCTIONS dataset. This strategy not only enhances the model's adaptability across a wide spectrum of multi-modal tasks but also boosts its performance in zero-shot learning scenarios. The alignment techniques employed by MULTIINSTRUCT, such as byte-pair encoding and VQ-GAN, play a crucial role in aligning text and image tokens within a unified vocabulary. This seamless integration allows the model to effectively process and interpret various types of inputs and outputs. The use of a unified sequence-to-sequence architecture facilitates a deeper integration and alignment of vision and language modalities, underscoring the model's innovative approach to bridging the gap between different types of data. The datasets used for training and fine-tuning, namely MULTIINSTRUCT and NATURAL INSTRUCTIONS, are specifically chosen to bolster the model's capabilities in handling multi-modal tasks and instructions, showcasing its versatility and effectiveness in enhancing multi-modal zero-shot learning.
</details> 

### **MouSi: Poly-Visual-Expert Vision-Language Models**

MouSi pushes the boundaries of VLMs by incorporating multiple visual experts like CLIP and SAM, utilizing a poly-expert fusion network to combine their outputs and interface with powerful LLMs like Vicuna, thereby enabling a more comprehensive understanding and processing of visual information.

[![arXiv](https://img.shields.io/badge/arXiv-2401.17221-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.17221) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/fudannlplab/mousi)  
Xiaoran Fan, Tao Ji, Changhao Jiang, Shuo Li, Senjie Jin, Sirui Song, Junke Wang, Boyang Hong, Lu Chen, Guodong Zheng, Ming Zhang, Caishuang Huang, Rui Zheng, Zhiheng Xi, Yuhao Zhou, Shihan Dou, Junjie Ye, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, Yu-Gang Jiang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/7e09c9d8-4c18-4970-9a24-b5e538285a72" alt="Architecture diagram for MouSi: Poly-Visual-Expert Vision-Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MouSi**: Represents an innovative approach to Vision-Language Models (VLMs) by integrating multiple visual experts into a unified architecture, aiming to surpass the limitations inherent to models reliant on a singular visual component. This architecture leverages a poly-expert fusion network, which incorporates outputs from varied visual experts, such as CLIP for image-text matching and SAM for image segmentation. This network facilitates an efficient interface with pre-trained Large Language Models (LLMs), notably utilizing a model like Vicuna v1.5. MouSi distinguishes itself by employing a multi-expert visual encoder that selects relevant experts from a pool, and it features two types of **poly-expert fusion networks: a projection fusion method and a Q-Former fusion method.** The training methodology of MouSi is characterized by a two-phase approach. Initially, during the pre-training phase, both the text-only LLM and the multi-expert encoder are kept static, with the training focus squarely on the poly-visual fusion network. Subsequently, in the fine-tuning phase, the LLM is activated for training in conjunction with the poly-visual fusion network, using high-quality supervised datasets. This methodology ensures that MouSi benefits from robust pre-existing language models while simultaneously enhancing its capability to process and integrate complex visual information. For alignment and fusion of the multimodal inputs, MouSi employs its poly-expert fusion network to amalgamate the outputs from the various visual experts, aligning them with the vision input tokens. This alignment is critical for encoding vision and text cohesively, a process facilitated by either the projection fusion method or the more complex Q-Former fusion method. These methods allow for the effective compression of multi-channel visual information into a format that can be efficiently processed alongside textual data. The datasets used in MouSi's training regimen include LCS-558K and the LAION-CC-SBU collection for pre-training, aimed at aligning text and image representation spaces, and diverse, high-quality SFT datasets for fine-tuning, enhancing the model's performance across a broad spectrum of multimodal tasks.
</details> 

### **LaVIN: Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models**

LaVIN offers an efficient and cost-effective approach to vision-language instruction tuning by employing a Mixture-of-Modality Adapter (MM-Adapter), significantly reducing trainable parameters and enabling a streamlined optimization process for LLMs without extensive pre-training.

[![arXiv](https://img.shields.io/badge/arXiv-2305.15023v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.15023v3) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/luogen1996/lavin)  
Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, Rongrong Ji
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/8afc8259-fa72-4e52-8080-a4ea12208e32" alt="Architecture diagram for LaVIN: Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LaVIN**: This model introduces the Mixture-of-Modality Adaptation (MMA) learning regime, a pioneering method that leverages **lightweight adapters** to fine-tune LLMs for vision-language (VL) instruction tasks. The core of LaVIN's architecture is the **Mixture-of-Modality Adapter (MM-Adapter)**, which connects the image encoder to the LLM using minimal adaptation modules, allowing for a streamlined optimization of the multimodal LLM through a relatively small number of parameters. The training methodology of LaVIN is notably efficient, employing the MMA strategy to fine-tune only the inserted adapters, thus significantly reducing the optimized parameter count to between three to five million. This method substantially lowers both training time and storage requirements, circumventing the need for additional VL pre-training. The MM-Adapter is instrumental in facilitating the seamless transition between single- and multi-modal instructions, thereby enhancing the model's adaptability to various VL tasks. Additionally, it employs a dynamic routing function that adjusts adaptations for input features, enabling an effective integration of vision and text embeddings. LaVIN's performance and versatility are further demonstrated through its application on diverse datasets, including ScienceQA, Alphaca-52k, and LLaVA-158k. ScienceQA is utilized to assess the model's multimodal question-answering capabilities, while the Alphaca-52k (text-only) and LLaVA-158k (text-image pairs) datasets are leveraged to refine and expand LaVIN's functionality as a multimodal chatbot. This strategic use of datasets underscores LaVIN's advanced vision-language understanding, illustrating its potential to significantly contribute to the field of multimodal learning and interaction.
</details> 

### **Nous-Hermes-2-Vision - Mistral 7B**

Nous-Hermes-2-Vision builds upon OpenHermes-2.5 by integrating the efficient SigLIP-400M vision encoder and incorporating a custom dataset with function calling capabilities, enabling it to not only understand visual and textual information but also extract specific text from images, advancing its functionality as a Vision-Language Action Model.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha)  
This project is led by qnguyen3 and teknium.
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Nous-Hermes-2-Vision**: Represents a notable advancement in the realm of Vision-Language Models, marking its distinction through the integration of two key enhancements that elevate its capabilities beyond traditional models. This model is an evolution from its predecessor, **OpenHermes-2.5-Mistral-7B**, and distinguishes itself by incorporating the **SigLIP-400M** for significantly improved performance and efficiency, moving away from the standard reliance on larger 3B vision encoders. Additionally, it introduces a custom dataset that includes function calling capabilities, transforming it into a more dynamic Vision-Language Action Model. The training of Nous-Hermes-2-Vision utilized a diverse dataset comprising 220K images from LVIS-INSTRUCT4V, 60K from ShareGPT4V, 150K private function calling data, and 50K conversations from teknium's OpenHermes-2.5. Such a varied dataset ensures the model's proficiency across a broad spectrum of vision-language tasks, including object recognition, instruction following, and conversational understanding. The model's innovative approach to integrating vision and language, particularly through the use of custom datasets for function calling, allows for encoding vision and text together in a way that supports action-oriented tasks and automation. A key feature of Nous-Hermes-2-Vision is its ability to interact with images to extract valuable text information from visual content, thus enabling detailed analyses and responses in natural language. This capability is underscored by the model's utilization of the SigLIP-400M, opting for a more lightweight and efficient architecture while enhancing performance in vision-language tasks. The model is further enriched with a custom dataset that includes **function calling**, allowing for the extraction of written information from images through specific tags, thus broadening its application scope for developers and researchers alike. Despite its innovative features, early usage of Nous-Hermes-2-Vision has revealed some challenges, such as hallucinations and spamming of EOS tokens. Recognizing these issues, the research team, led by Quan Nguyen and Teknium, has committed to releasing an updated version to address these problems, demonstrating their dedication to refining the model's capabilities.
</details> 

### **TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones**

TinyGPT-V prioritizes efficiency in multimodal large language models by combining a compact EVA-ViT visual encoder with linear projection layers and the powerful Phi-2 language model, achieving robust performance in vision-language tasks despite its smaller size.

[![arXiv](https://img.shields.io/badge/arXiv-2312.16862v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.16862v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/DLYuanGod/TinyGPT-V) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/llizhx/TinyGPT-V)  
Zhengqing Yuan, Zhaoxu Li, Lichao Sun
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/3e7c93bc-7963-4c2e-b207-226a03d152ca" alt="Architecture diagram for TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**TinyGPT-V**: introduces a compact yet powerful architecture tailored for efficient multimodal large language model applications, leveraging small backbones for streamlined processing. This model integrates a visual encoder, specifically EVA of Vision Transformer (ViT), with **linear projection layers** and the Phi-2 language model, constituting its core components. The visual encoder remains inactive during training, focusing on image resolution adjustments across various stages to enhance image understanding. The **linear projection layers**, particularly with the incorporation of the **Q-Former layer** from BLIP-2, aim to efficiently embed visual features into the language model, reducing the number of parameters needing training. The Phi-2 large language model backbone, a 2.7 billion-parameter model, excels in reasoning and language comprehension, effectively handling vision-language operations including spatial location tasks through textual bounding box depictions. The training of TinyGPT-V unfolds across four stages: warm-up, pre-training, instruction fine-tuning, and multi-task learning. Each stage is meticulously designed to progressively enhance the model's capabilities in understanding and generating language based on visual inputs, with a special emphasis on human-like learning and conversation abilities in later stages. The use of datasets such as LAION, CC3M, SBU, and more, across these stages, supports the model's development in vision-language understanding, generation, and task execution like visual question answering and image captioning. A noteworthy aspect of TinyGPT-V's architecture is the implementation of normalization techniques and LoRA (Low-Rank Adaptation) to stabilize training and optimize the model's performance across different modalities. Addressing challenges like NaN or INF values in multimodal data computation, these mechanisms enhance training stability and efficiency. Furthermore, the model employs a multi-task instruction template to manage task ambiguity, utilizing MiniGPT-v2 tokens for task-specific instructions, facilitating precise and accurate task execution.
</details> 

### **CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding**

CoVLM distinguishes itself by using novel communication tokens to enable dynamic interaction between its CLIP ViT-L image encoder, YOLOX detection network, and Pythia language model, facilitating sophisticated communication for superior compositional reasoning in vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03354v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03354v1)  
Junyan Li, Delin Chen, Yining Hong, Zhenfang Chen, Peihao Chen, Yikang Shen, Chuang Gan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/80e807cb-c2cf-491a-a3b4-1223afde1981" alt="Architecture diagram for CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CoVLM**: This model is distinct in its approach, employing a novel set of **communication tokens** that facilitate dynamic interaction between a vision encoder, detection network, and a language model (LLM). The architecture of CoVLM integrates a CLIP ViT-L image encoder and a YOLOX detection network, alongside a pre-trained Pythia model for language processing. These components work in tandem to guide the LLM in composing visual entities and relationships within the textual context, enhancing the model's ability to dynamically communicate with the vision encoder and detection network. CoVLM is pre-trained on a diverse and extensive image-text dataset comprising 97 million image-text pairs, drawn from a variety of sources. This extensive dataset supports the model's grounding pipeline, which is crucial for associating text spans with their corresponding visual entities in images. The model utilizes special communication tokens for facilitating iterative communication between its vision and language components, enabling a sophisticated form of top-down and bottom-up communication. This communication is key to achieving high performance in vision-language tasks, as it allows the model to seamlessly integrate and interact between language tokens and visual embeddings. The datasets employed for pre-training, such as COCO, CC3M, CC12M, Visual Genome, SBU, and LAION400M, are meticulously selected to enhance the model's ability to ground image-text pairs effectively. This strategic choice is aimed at facilitating the association of textual descriptions with their corresponding visual entities, thereby improving the model's overall performance across a range of multimodal tasks. CoVLM's innovative approach to integrating visual detection networks with LLMs enables a new level of compositional reasoning, setting it apart from previous vision-language models.
</details> 

### **GLaMM: Pixel Grounding Large Multimodal Model**

GLaMM excels in pixel-level grounding by utilizing a five-component architecture encompassing global and regional image encoders, an LLM, a grounding image encoder, and a pixel decoder, allowing for comprehensive visual understanding and precise object localization within images.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03356-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03356) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mbzuai-oryx/groundingLMM)  
Hanoona Rasheed, Muhammad Maaz, Sahal Shaji Mullappilly, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Erix Xing, Ming-Hsuan Yang, Fahad S. Khan  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/ccb22206-6a48-4b77-8cc1-094fe86d72fd" alt="Architecture diagram for GLaMM: Pixel Grounding Large Multimodal Model" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**GLaMM**: At its core, GLaMM comprises five essential components: the **Global Image Encoder, Region Encoder, Language Model (LLM), Grounding Image Encoder, and Pixel Decoder**. This architecture is designed to facilitate a wide range of interactions with visual content, from scene-level understanding through the Global Image Encoder, to detailed region-level interpretations via the Region Encoder, and down to precise pixel-level object grounding with the Grounding Image Encoder. The Pixel Decoder component further enriches the model's capabilities by generating **segmentation masks**, enabling GLaMM to respond to both textual and visual prompts with high fidelity. The training methodology of GLaMM involves a dual-pathway approach, encompassing both automated and manual data annotation pipelines to create the Grounding-anything Dataset (GranD). GranD is pivotal for the model's training, especially for its Grounded Conversation Generation (GCG) task, offering a rich set of 7.5 million unique concepts grounded in 810 million regions, complete with segmentation masks. This dataset not only supports the pretraining and fine-tuning phases of GLaMM but also underlines its unique ability to generate grounded conversations that are contextually relevant to the visual stimuli. Alignment techniques within GLaMM utilize a vision-to-language (V-L) projection layer, facilitating the mapping of image features into the language space, thereby ensuring effective text-image alignment. Furthermore, the model employs a language-to-prompt (L-P) projection layer, transforming text embeddings related to segmentation into the decoder space. This dual-projection system allows for an integrated encoding of vision and text, bolstering GLaMM's capacity for pixel-level grounding and positioning it as a significant advancement in the field of multimodal interactions.
</details> 

### **COSMO: COntrastive Streamlined MultimOdal Model with Interleaved Pre-Training**

COSMO presents a streamlined multimodal framework by combining a Vision Transformer with a partitioned Large Language Model, optimizing the processing of interleaved data sequences through a combination of language modeling and contrastive loss functions.

[![arXiv](https://img.shields.io/badge/arXiv-2401.00849v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.00849v1) [![Website](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://fingerrec.github.io/cosmo)

Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, Mike Zheng Shou
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0c256daa-1573-4110-a665-5927ee2e293f" alt="Architecture diagram for COSMO: COntrastive Streamlined MultimOdal Model with Interleaved Pre-Training" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**COSMO**: This framework is distinctive for its architecture that merges a visual encoder, leveraging the Vision Transformer (ViT) from Open-CLIP, with a partitioned Large Language Model (LLM). The LLM is systematically divided into segments dedicated to unimodal text processing and multimodal data handling, aiming to streamline the overall processing of interleaved data sequences. The introduction of an additional contrastive loss component stands out as a strategy to improve performance across both classification and generation tasks. Training of COSMO is carried out through a unique combination of language modeling loss and contrastive loss, focusing on the efficient management of interleaved text and visual sequences. This process is optimized with the use of the AdamW optimizer, a cosine learning rate schedule, and the implementation of DeepSpeed fp16 precision, distributed across 128 NVIDIA V100 GPUs. The partitioning strategy of the LLM into dedicated components is a testament to the framework's commitment to computational efficiency and efficacy in handling extensive data sequences. The model's alignment techniques are notably advanced, featuring a learnable query that facilitates global attention across all tokens, alongside an additional query for **Text Fusion Layers**, optimizing the model's understanding of token sets and enhancing image-text alignment through contrastive loss. **The gated cross-attention layers** for multimodal fusion introduce a significant reduction in learnable parameters by introducing bottlenecks in input and output feature channels. This method of lightweight fusion is pivotal in integrating visual information for precise next-token prediction. COSMO's training leverages a diverse array of datasets including CC3M, SBU, LAION400M, DataComp1B, MMC4, WebVid, and Howto-Interlink7M. The introduction of Howto-Interlink7M, in particular, underscores the model's innovative approach to improving video-language understanding through high-quality annotated captions, demonstrating its effectiveness across 14 diverse downstream tasks.
</details> 

### **FireLLaVA**

FireLLaVA breaks new ground by combining the CodeLlama 34B Instruct model for advanced language understanding with a CLIP-ViT-based visual interpretation component, training on a unique dataset incorporating bounding box labels and captions to excel in visual language conversations.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/fireworks-ai/FireLLaVA-13b)   

<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FireLLaVA**: As the first of its kind within the LLaVA lineage, FireLLaVA integrates a dual-component architecture that leverages the CodeLlama 34B Instruct model for nuanced language understanding and a visual interpretation component akin to OpenAI's CLIP-ViT. This model is distinctive for its use of bounding box labels and captions to generate visual language conversations, a method that underscores its innovative approach to multi-modal training. The training regimen for FireLLaVA is meticulously crafted, utilizing 588K lines of visual question answering and conversation data. This dataset amalgamates permissive original LLaVA data with newly generated data from Fireworks.ai, demonstrating a unique approach to instruction fine-tuning that enhances the model's ability to comprehend and articulate responses that bridge textual and visual inputs. The integration of bounding box labels and captions not only serves as a mechanism for generating training data but also facilitates the alignment of text and image data, a crucial step in achieving coherent multi-modal understanding. Although the specific methods employed for alignment fusion within FireLLaVA's architecture remain under-described, it is inferred that embedding fusion plays a critical role in synthesizing vision and text inputs. By drawing on original LLaVA training materials and Fireworks.ai's proprietary data, FireLLaVA sets a precedent for the development of VLMs capable of navigating the complexities of commercial applications. This model embodies a significant advancement in the field of visual language modeling, offering insights into the potential of OSS models to contribute to the evolving landscape of multi-modal AI research and deployment.
</details> 

### **u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model**

u-LLaVA introduces a novel projector-based architecture that unifies multi-modal tasks by connecting specialized expert models with a central Large Language Model (LLM), enabling seamless modality alignment and efficient multi-task learning through a two-stage training approach.

[![arXiv](https://img.shields.io/badge/arXiv-2311.05348-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.05348) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/OPPOMKLab/u-LLaVA)  
Jinjin Xu, Liwu Xu, Yuzhe Yang, Xiang Li, Yanchun Xie, Yi-Jie Huang, Yaqian Li
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/dcb6b046-fa56-4a02-9123-2ef2185c635a" alt="Architecture diagram for u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**u-LLaVA**: Represents a pioneering approach in the integration of Large Language Models (LLMs) with specialized expert models to address a wide array of multi-modal tasks. This architecture is designed to leverage the strengths of LLMs as a central hub, facilitating seamless modality alignment and multi-task learning. Through a novel **projector-based structure** that incorporates CLIP's Vision Transformer (ViT-L/14) and LLaMA2, u-LLaVA introduces a flexible framework capable of handling diverse modalities and tasks. The system integrates special tokens for modality and task expressions, alongside dedicated modules for segmentation, grounding, and in-painting, to enrich its multi-modal capabilities. The training methodology of u-LLaVA is executed in two distinct stages, beginning with a coarse-grained alignment to ensure the alignment of representation spaces across different modalities. This foundational step is crucial for establishing a common ground for further, more nuanced task-specific adaptations. Following this, a fine-grained alignment phase focuses on the refinement of task-specific instruction data, optimizing the model's performance for targeted applications. This dual-stage training approach ensures that u-LLaVA can efficiently adapt to a variety of tasks with minimal additional training requirements. Central to u-LLaVA's effectiveness is its innovative use of projector-based alignment techniques and fusion methods, which enable the integration of visual and textual representations within the LLM's framework. By mapping hidden states and text embeddings through projectors, u-LLaVA facilitates modality fusion, leveraging the extensive knowledge embedded within LLMs for complex task solving. The datasets utilized for training, including LLaVA CC3M, Conversation-58K, Detail-23K, and others, are meticulously curated to support the model's versatile capabilities across tasks such as image captioning, video captioning, visual question answering (VQA), referential expression comprehension (RES), semantic segmentation, and salient object detection/segmentation. This strategic selection and organization of datasets underscore u-LLaVA's commitment to advancing multi-modal task unification through Large Language Models.
</details> 

### **MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**

MoE-LLaVA introduces a novel approach by incorporating Mixture of Experts (MoE) within a large vision-language model, using learnable routers to selectively activate expert modules for processing specific tokens, thereby enhancing efficiency and enabling nuanced understanding of multimodal inputs.

[![arXiv](https://img.shields.io/badge/arXiv-2401.15947-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.15947) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/PKU-YuanGroup/MoE-LLaVA) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LanguageBind/MoE-LLaVA)  
Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Munan Ning, Li Yuan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5e214b-be64-4aac-aba4-04c97970b9de" alt="Architecture diagram for MoE-LLaVA: Mixture of Experts for Large Vision-Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MoE-LLaVA**: Represents an innovative leap in the development of large vision-language models through the integration of **Mixture of Experts (MoE)** within a sophisticated architectural framework. This model is characterized by its sparse design, wherein individual tokens are directed towards a selection of experts based on **learnable routers**, ensuring that only the top-k experts are activated for any given token's processing. Such an approach not only enhances the model's efficiency but also its capability to handle diverse and complex data inputs by leveraging specialized processing paths for different types of information. At the heart of MoE-LLaVA's architecture are several critical components, including a vision encoder, **a visual projection MLP layer**, **word embedding layers**, **multi-head self-attention blocks**, **feed-forward neural networks**, and notably, **the MoE blocks** themselves. These elements are seamlessly integrated through the use of layer normalization and residual connections, establishing a robust and adaptable framework capable of deep multimodal understanding. The training methodology for MoE-LLaVA is meticulously structured in three stages, each designed to gradually enhance the model's proficiency in integrating and processing visual and textual data. This includes initial adaptation of image tokens, training of all LLM parameters excluding the vision encoder, and specialized training of the MoE layers, with the latter utilizing initialization weights from previous stages for optimal performance. Alignment techniques and fusion methods employed by MoE-LLaVA are pivotal in achieving a harmonious integration of text and image modalities. By utilizing learnable routers to dynamically allocate tokens to the most apt experts and subsequently processing these through a combination of LLM and MoE blocks, the model achieves a nuanced understanding of multimodal inputs. The datasets employed throughout the training phases—ranging from LLaVA-PT for pretraining to Hybrid-FT for multimodal instruction tuning, and LLaVA-FT for fine-tuning the MoE layers—further underscore the model's ability to refine its understanding across a broad spectrum of multimodal tasks. This strategic deployment of diverse datasets not only facilitates a comprehensive tuning of the model's capabilities but also underscores its potential in advancing the field of vision-language processing.
</details> 

### **BLIVA: A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions**

BLIVA augments the InstructBLIP model with a Visual Assistant, incorporating encoded patch embeddings alongside learned query embeddings to enhance the LLM's understanding of text-rich visual contexts, thereby excelling in handling complex visual questions.

[![arXiv](https://img.shields.io/badge/arXiv-2308.09936v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.09936v3) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mlpc-ucsd/bliva)  
Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/44c53b8a-ad35-4eca-a68b-63af32e6ccf1" alt="Architecture diagram for BLIVA: A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BLIVA**: This model builds upon the foundation of InstructBLIP, incorporating a Visual Assistant to enhance its understanding and processing of text-rich visual contexts. BLIVA's architecture is designed to capture the intricacies of visual content that may be overlooked during the query decoding process by melding learned query embeddings from InstructBLIP with directly projected encoded patch embeddings. The core components of BLIVA include a vision tower, responsible for encoding visual inputs into patch embeddings; a **Q-former**, which refines query embeddings; and a **projection layer** that bridges the visual and linguistic domains, enabling the LLM to access a rich tapestry of visual knowledge. The training methodology of BLIVA is structured around a two-stage scheme: initial pre-training on image-text pairs derived from captioning datasets, followed by instruction tuning using Visual Question Answering (VQA) data. This process begins with the pre-training of the projection layer for patch embeddings, succeeded by the fine-tuning of both the Q-former and the projection layer, while the image encoder and LLM remain static to prevent catastrophic forgetting. This approach ensures that BLIVA is finely attuned to visual information, enhancing its ability to handle complex visual questions. BLIVA's alignment techniques and fusion methods stand out for their integration of learned query embeddings with an additional visual assistant branch that utilizes encoded patch embeddings. By concatenating these embeddings and feeding them directly into the LLM, BLIVA significantly improves the model's text-image visual perception capabilities. This enhanced multimodal understanding is further demonstrated through the use of diverse datasets, including image captioning datasets for pre-training, instruction tuning VQA data for performance enhancement, and YTTB-VQA (YouTube Thumbnail Visual Question-Answer pairs) to showcase BLIVA's proficiency in processing text-rich images and its suitability for real-world applications.
</details> 

### **MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices**

MobileVLM offers a mobile-optimized vision-language model that combines a CLIP ViT-L/14 visual encoder with the efficient MobileLLaMA language model and a Lightweight Downsample Projector (LDP), enabling effective multimodal processing and alignment within the constraints of mobile devices.

[![arXiv](https://img.shields.io/badge/arXiv-2312.16886-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.16886) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/meituan-automl/mobilevlm)  
Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/59a06109-ba49-4299-951c-d7c0c562bca3" alt="Architecture diagram for MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MobileVLM**: Introduces a compact yet robust architecture designed to facilitate efficient vision-language tasks on mobile devices, distinguishing itself through a blend of specialized components and a streamlined training methodology tailored for edge computing environments. At its core, MobileVLM integrates a visual encoder based on the CLIP ViT-L/14 model with a resolution of 336x336, MobileLLaMA—a language model optimized for mobile devices, and a **Lightweight Downsample Projector (LDP)** that bridges the gap between visual and textual data with minimal computational overhead. This synergy between components ensures that MobileVLM can process and align multimodal inputs effectively, making it well-suited for mobile applications where resource efficiency is paramount. The training regimen for MobileVLM unfolds in three distinct phases, each contributing uniquely to the model's development. Initially, the language model undergoes pre-training using the text-centric RedPajama v1 dataset, laying a solid linguistic foundation. Subsequent supervised fine-tuning leverages multi-turn dialogues between humans and ChatGPT, refining the model's conversational abilities. The final stage involves training the integrated vision-language model on diverse multimodal datasets, equipping MobileVLM with the capacity to interpret and respond to both visual and textual stimuli. This comprehensive training approach ensures that MobileVLM achieves a balance between performance and efficiency, making it adept at handling complex vision-language interactions on mobile platforms. Central to MobileVLM's effectiveness is the Lightweight Downsample Projector (LDP), a novel component designed for the efficient alignment of visual and textual features. By employing mobile-friendly operations such as depth-wise convolution, LDP manages to downsample visual tokens to match the language model's input dimensions, preserving spatial information while minimizing computational demands. This alignment mechanism, in conjunction with the efficient fusion of vision and text embeddings, enables MobileVLM to maintain high levels of accuracy and responsiveness in mobile environments. Through the use of carefully selected datasets, including RedPajama v1 for linguistic pre-training and various multimodal datasets for comprehensive vision-language modeling, MobileVLM showcases its capability to navigate the challenges of mobile-based vision-language tasks with remarkable efficiency.
</details> 

### **FROZEN: Multimodal Few-Shot Learning with Frozen Language Models**

FROZEN enables multimodal few-shot learning by pairing a pre-trained, frozen language model with a trainable vision encoder (NF-ResNet-50) that converts images into a dynamic visual prefix, allowing the model to process and generate language in context with visual data without altering its core language capabilities.

[![arXiv](https://img.shields.io/badge/arXiv-2106.13884-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2106.13884)  
Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/4156475d-e501-495e-98bb-66efdd5b03f7" alt="Architecture diagram for FROZEN: Multimodal Few-Shot Learning with Frozen Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FROZEN**: Presents an innovative approach to extending the few-shot learning capabilities of pre-existing language models into the multimodal domain, specifically targeting the integration of visual and linguistic elements without the need to alter the foundational language model parameters. This methodology introduces a vision encoder, specifically an **NF-ResNet-50**, designed to translate images into a continuous sequence of embeddings. These embeddings serve as a visual prefix to the input for a pre-trained autoregressive language model based on the Transformer architecture, enabling the language model to process and generate content relevant to the given visual context. The core innovation lies in the system's modularity, achieved by keeping the language model's weights static while **only updating the vision encoder** during training. This approach leverages the Conceptual Captions dataset, focusing on the alignment of image-caption pairs to train the vision encoder, thereby simplifying the integration of visual data into language models. The architecture of FROZEN is distinguished by its use of a dynamic visual prefix, a departure from the conventional static text prompts typical in prefix tuning. This dynamic prefix is achieved by linearly mapping and reshaping the vision encoder's output into a sequence of embeddings, mirroring the functionality of text-based prefix tokens in traditional language model tuning. This mechanism allows the model to adapt more fluidly to multimodal inputs, enhancing its ability to interpret and generate language that is contextually aligned with visual data. The employment of a dynamic visual prefix is a key factor in FROZEN's ability to improve task performance across multimodal settings through in-context learning, providing a novel solution to the challenge of incorporating visual information into the language generation process. The utilization of the Conceptual Captions dataset is central to FROZEN's training methodology, enabling the **vision encoder to adeptly convert images** into a format that the language model can process. This dataset serves the dual purpose of enhancing the model's understanding of visual content and its associated linguistic descriptions, thereby facilitating the generation of accurate and contextually relevant captions. The strategic combination of a static language model with a trainable vision encoder encapsulates FROZEN's approach to multimodal few-shot learning, offering a streamlined and effective pathway to integrating visual data into linguistic models.
</details> 

### **Flamingo: a Visual Language Model for Few-Shot Learning**

Flamingo pioneers a Perceiver-based VLM architecture that utilizes a Perceiver Resampler and gated cross-attention dense layers, enabling it to process interleaved text and visual sequences for impressive few-shot learning performance across a variety of multimodal tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2204.14198v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2204.14198v2)  
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/b46ebf3e-67fc-401e-a6ea-6f4797da372d" alt="Architecture diagram for Flamingo: a Visual Language Model for Few-Shot Learning" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Flamingo**: Represents an innovative approach in the realm of Visual Language Models (VLMs), specifically designed to excel in few-shot learning tasks. This model is distinguished by its capacity to process sequences of text tokens that are interwoven with visual data, such as images or videos, to generate textual outputs. At the core of Flamingo's architecture is the adoption of a Perceiver-based framework that adeptly manages high-resolution visual inputs. This design choice enables the handling of complex, multimodal information streams by transforming large visual feature maps into a concise number of visual tokens through the **Perceiver Resampler**. Further refining its architecture, Flamingo incorporates **gated cross-attention dense (GATED XATTN-DENSE) layers**, which play a pivotal role in conditioning the language model on visual inputs, thereby facilitating a nuanced understanding and generation of language based on the visual context. The training regimen of Flamingo is both extensive and diverse, encompassing a wide array of datasets culled from the web. This includes a rich mixture of interleaved image and text data, image-text pairs, and video-text pairs, which collectively contribute to the model's robust few-shot learning capabilities. A distinctive aspect of Flamingo's training is its strategy to minimize a weighted sum of per-dataset expected negative log-likelihoods of text given visual inputs. This approach, combined with a gradient accumulation strategy across all datasets, ensures comprehensive learning from varied multimodal contexts. The datasets employed in training, namely MultiModal MassiveWeb (M3W), ALIGN dataset, Long Text & Image Pairs (LTIP), and Video & Text Pairs (VTP), each serve a specific purpose. M3W facilitates training on interleaved text and image data, ALIGN on image-text pairs, LTIP on high-quality image-text pairs, and VTP on video-text pairs, ensuring Flamingo's adeptness across different visual language tasks. In its alignment techniques, Flamingo introduces an image-causal modeling approach to manage text-to-image cross-attention effectively, allowing the model to attend selectively to visual tokens of the image that immediately precede the given text token in the sequence. This capability is further enhanced by the gated cross-attention layers, which employ a tanh-gating mechanism to merge the output of these layers with the input representation from the residual connection. Such an alignment fusion method ensures that Flamingo can seamlessly integrate vision and text embeddings, underscoring its innovative architecture and the breadth of its training. Through these mechanisms, Flamingo stands out as a significant advancement in the integration of visual and textual data for language model training, showcasing its versatility and effectiveness in few-shot learning scenarios.
</details> 

### **OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models**

OpenFlamingo, an open-source adaptation of DeepMind's Flamingo, combines a CLIP ViT-L/14 visual encoder with a 7B parameter language model, utilizing frozen cross-attention modules for efficient and effective multimodal fusion during the decoding process, resulting in impressive performance on various vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2308.01390-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2308.01390) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/mlfoundations/open_flamingo)  
Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**OpenFlamingo**: Represents an innovative leap in the integration of vision and language models, providing an open-source adaptation of DeepMind's Flamingo framework. This model is structured around a powerful combination of a CLIP Vision Transformer Large (ViT-L/14) for encoding visual inputs and a 7-billion parameter Multimodal Pretrained Transformer (MPT-7B) for language processing. The architecture is distinctive for its inclusion of cross-attention modules within every fourth decoder block of the language model, which remains frozen during training. These modules are pivotal for the model's ability to attentively merge visual information with textual context during the decoding process, thereby enhancing its multimodal understanding. The training methodology for OpenFlamingo is grounded in a comprehensive strategy that harnesses the vast data landscape of the internet. It utilizes a rich dataset amalgam comprising LAION-2B and the Multimodal version of the Common Crawl (C4) dataset, focusing on image-text pair sequences. This approach is facilitated by DistributedDataParallel training across an impressive array of 64 A100 80GB GPUs, leveraging automatic BF16 mixed precision for optimized performance. The model's alignment techniques are inspired by the original Flamingo's design philosophy, which emphasizes the importance of keeping the core vision and language models static while dynamically training the connecting **cross-attention modules** for decoding. This selective training process ensures that OpenFlamingo can effectively fuse visual and textual data, thereby significantly improving its proficiency in generating relevant text based on visual cues. Furthermore, the datasets used are instrumental in refining OpenFlamingo's capacity for understanding complex visual-textual interactions. Trained specifically on image-text sequences, the model demonstrates superior performance in tasks requiring nuanced interpretation of visual content, such as captioning, visual question answering, and image classification. This strategic focus on multimodal datasets underscores the model's purpose to bridge the gap between visual perception and linguistic expression, marking a substantial advancement in the field of multimodal AI. Through these architectural innovations and training strategies, OpenFlamingo sets a new standard for open-source models in the domain of visual-language tasks.
</details> 

### **IDEFICS**

IDEFICS, an 80B parameter vision-language model inspired by Flamingo, processes interleaved image and text sequences, utilizing a GPT-4 and Flamingo-based architecture to achieve robust multimodal understanding, trained on a diverse range of web-based datasets, including the specialized OBELICS dataset.

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HuggingFaceM4/idefics-80b)
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**IDEFICS**: stands for "an 80 billion parameters vision and language model," distinguishing itself as a robust model designed to mimic Flamingo's capabilities while integrating substantial advancements in handling multimodal inputs. This model is crafted to accept sequences of images and text, generating text outputs that reflect a deep understanding of both visual and textual information. The architecture of IDEFICS builds on the foundations laid by GPT-4 and Flamingo, showcasing a harmonious blend of vision and language processing capabilities within a singular model framework. This strategic design allows IDEFICS to process and interpret complex multimodal inputs efficiently, setting a new precedent in the field of integrated vision-language models. During its development, IDEFICS faced challenges related to loss spikes, which were effectively mitigated through rollback strategies and precise adjustments in the learning rate. An auxiliary z-loss was introduced to normalize logits, significantly enhancing training stability. The model adopts Flamingo's methodological approach for alignment, utilizing pretrained vision and language backbones to foster a nuanced cross-modal understanding. Although specific details on fusion techniques for vision and text embeddings remain under wraps, it is inferred that the model employs **cross-attention mechanisms** akin to Flamingo's, facilitating a sophisticated integration of visual and textual data. Training on OBELICS—a meticulously curated collection of interleaved image-text web documents—and other web-scraped datasets, IDEFICS aims to excel in multimodal tasks. The OBELICS dataset, in particular, is designed to augment the model's performance by providing access to longer text contexts and a diverse array of web document types. This strategic dataset selection underscores IDEFICS's commitment to enhancing its proficiency across a spectrum of multimodal applications, leveraging the rich, varied content found in web documents to refine its understanding and output generation capabilities.
</details> 

### **PaLI: A Jointly-Scaled Multilingual Language-Image Model**

PaLI distinguishes itself as a jointly-scaled multilingual language-image model that utilizes a unified interface to process both unimodal and multimodal tasks, integrating a powerful ViT-e visual encoder with an mT5-based text encoder-decoder Transformer for comprehensive language and vision understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2209.06794-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2209.06794) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/big_vision)  
Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2565afb0-901c-4438-9488-c73a86261aa5" alt="Architecture diagram for PaLI: A Jointly-Scaled Multilingual Language-Image Model" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PALI**: This model stands out by its ability to handle both unimodal (language or vision) and multimodal (language and vision together) tasks through a unified interface that accepts images and text as inputs, subsequently generating text as the output. The architecture of PALI ingeniously integrates a text encoder-decoder Transformer, based on pre-trained mT5 models, with visual tokens processed by a Vision Transformer (ViT) named ViT-e. ViT-e marks a significant advancement in visual processing with up to 4 billion parameters, setting a new precedent for the integration of visual components within language models. The PALI model utilizes pre-trained unimodal checkpoints, optimizing the efficiency of its training processes. Training methodologies for PALI are robust and diverse, incorporating a mixture of pre-training tasks aimed at enhancing the model's capability across a broad spectrum of downstream applications. Leveraging the expansive image-language dataset WebLI, which encompasses 10 billion images and texts across over 100 languages, PALI undergoes a comprehensive two-phase training regime. This includes a specific focus on high-resolution training for its largest model variant, PALI-17B. Such an approach ensures that PALI is not just multilingual but also highly adept at processing and understanding complex visual and textual data. The alignment and fusion techniques employed by PALI are particularly noteworthy. By adopting a unified modeling interface, the model treats various tasks with a task-agnostic perspective, allowing it to seamlessly transition between different types of vision and language tasks. The fusion of vision and text is achieved through **a cross-attention mechanism**, where a sequence of visual tokens from the Vision Transformer is integrated with the text encoder-decoder Transformer. This method enables an efficient and effective blending of multimodal information. The use of datasets such as WebLI, Conceptual Captions, and OCR data from WebLI, along with others like VQ2A-CC3M and Open Images, further enriches PALI's training, equipping it with a vast and versatile multimodal proficiency. This proficiency spans across multilingual settings, captioning, OCR, and visual question answering (VQA), ensuring PALI's comprehensive understanding and generation capabilities across a wide array of languages and tasks.
</details> 

### **PaLI-3 Vision Language Models: Smaller, Faster, Stronger**

PaLI-3 presents a powerful yet efficient vision-language model that integrates a contrastively pretrained 2B SigLIP vision model with a 3B UL2 Transformer, achieving impressive performance in tasks like captioning and visual question answering through a multi-stage training process that emphasizes scalability and robustness.

[![arXiv](https://img.shields.io/badge/arXiv-2310.09199-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.09199) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/kyegomez/PALI3)  
Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/92d34b30-b13b-44ed-90b5-3c8568a9b634" alt="Architecture diagram for PaLI-3 Vision Language Models: Smaller, Faster, Stronger" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PaLI-3** :Its architecture integrates a contrastively pretrained 2B **SigLIP vision model** with a 3B encoder-decoder UL2 Transformer, focusing on the efficient processing of visual and textual data. The training methodology of PaLI-3 includes **contrastive pretraining of the image encoder** on a vast scale of image-text data, subsequent multimodal training, and resolution increase stages to refine its performance further. These stages ensure that PaLI-3 achieves a nuanced understanding of visually-situated text and object localization, supported by datasets such as Web-scale image-text data, RefCOCO, WebLI, CC3M-35L, and various VQA datasets. The visual component of PaLI-3 utilizes a vision transformer pretrained in a contrastive manner, emphasizing efficiency, scalability, and robustness. This approach allows for a more nuanced pretraining of the image embedding component, which, when combined with text embeddings, enhances the model's ability to understand and generate text based on visual inputs. The full model employs these visual tokens alongside embedded input text tokens within a UL2 encoder-decoder framework, demonstrating its capability in generating text outputs for tasks such as captioning and visual question answering (VQA). PaLI-3's training process involves several key stages, starting with unimodal pretraining of the image encoder using image-text pairs from the web. This is followed by multimodal training, where the image encoder and text encoder-decoder are combined and trained on a mixture of tasks and data, focusing on visually-situated text and object detection. The resolution increase stage further enhances performance by fine-tuning the model with high-resolution inputs. Finally, task specialization involves fine-tuning PaLI-3 on individual benchmark tasks, optimizing its performance across a wide range of applications. 
</details> 

### **PaLM-E: An Embodied Multimodal Language Model**

PaLM-E innovates by embedding continuous sensory data, including images and sensor readings, into the language representation space of a pre-trained PaLM model, enabling it to process and generate text that reflects embodied reasoning and understanding of the physical world.

[![arXiv](https://img.shields.io/badge/arXiv-2303.03378-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.03378) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://palm-e.github.io)  
Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/67e5bbc7-1800-46e8-8ef1-b3b72a901a12" alt="Architecture diagram for PaLM-E: An Embodied Multimodal Language Model" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**PaLM-E**: Represents an innovative step in the development of multimodal language models by integrating continuous embodied observations—ranging from images and state estimates to various sensor modalities—into the linguistic embedding space of a pre-trained language model. It utilizes a decoder-only large language model (LLM) architecture that generates textual completions autoregressively, taking multimodal inputs into account. The core architecture of PaLM-E leverages a pre-trained PaLM as its language backbone, enhancing it with encoders that transform sensor modalities into a **sequence of vectors** compatible with the language model's embedding dimensions. This integration allows for the seamless combination of continuous sensor information with textual data, crafting multimodal sentences that the model processes. Training methodologies for PaLM-E are comprehensive and end-to-end, utilizing datasets composed of both continuous observations and textual information. The model employs a cross-entropy loss function for non-prefix tokens, with a training regimen that includes pre-trained Vision Transformers (ViTs) for image feature extraction alongside novel and pre-trained input encoders. The approach allows for flexibility in model training, including options for freezing pre-trained components or co-training them across varied data sets. This strategy ensures that PaLM-E benefits from both the depth of pre-trained models and the specificity of tailored encoders for continuous data. PaLM-E's alignment techniques and fusion methods are pivotal for its operation, employing encoders to integrate continuous sensor data into the linguistic embedding space effectively. This integration facilitates an understanding and generation of responses that reflect a blend of textual and sensor input, mimicking embodied reasoning. The model processes multimodal sentences—interleaved sequences of sensor observations and text—through its **self-attention layers**, similar to how it handles traditional text tokens. This methodology ensures a cohesive encoding of vision and text information. PaLM-E's training leverages a diverse array of datasets, including large-scale vision-and-language data and specialized robotics tasks datasets, aiming to excel across a broad spectrum of embodied reasoning tasks. This diverse training background enables PaLM-E to harness cross-domain transfer learning, enhancing its capabilities in specific robotics applications and general vision-language tasks alike.
</details> 

### **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**

MiniGPT-4 seamlessly blends visual and language processing by connecting a pretrained Vision Transformer and Q-Former to a frozen Vicuna LLM using a single linear projection layer, achieving impressive vision-language understanding through a two-stage training approach focused on efficient alignment and enhanced generation quality.

[![arXiv](https://img.shields.io/badge/arXiv-2304.10592v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.10592v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/vision-cair/minigpt-4)  
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/0e5ff945-1271-4189-8dd9-b0abd88eacc1" alt="Architecture diagram for MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MiniGPT-4**: presents an advanced integration of vision and language processing capabilities through a meticulously designed architecture that marries a frozen visual encoder with a frozen advanced Large Language Model (LLM), specifically Vicuna. At the heart of MiniGPT-4 is its novel approach to aligning visual and linguistic modalities: it employs **a single linear projection layer** to bridge the pretrained Vision Transformer (ViT) and **Q-Former** with the Vicuna LLM. This design choice underscores a commitment to efficiency, focusing on leveraging existing, robust components to achieve a seamless integration of visual features with sophisticated language capabilities. The training methodology for MiniGPT-4 is bifurcated into two distinct stages, optimizing both the initial alignment of visual and language features and the subsequent enhancement of generation reliability and naturalness. Initially, MiniGPT-4 undergoes training for 20,000 steps with a batch size of 256 on 4 A100 GPUs, utilizing a combined dataset from sources like Conceptual Captions, SBU, and LAION for foundational vision-language knowledge. This stage is crucial for establishing the basic alignment between the visual encoder and the Vicuna LLM. The second stage of finetuning, leveraging a curated dataset of 3,500 detailed image descriptions, is pivotal for refining the model's output, focusing on generating more detailed, reliable, and naturally flowing text. The strategic use of datasets in MiniGPT-4's training regimen underscores its dual objectives: foundational vision-language alignment and the enhancement of output naturalness and detail. Initial datasets facilitate the basic integration of visual and linguistic elements, while the curated dataset of detailed image descriptions serves to significantly improve the model's capability in generating nuanced and accurate natural language descriptions. Through this comprehensive and staged training approach, MiniGPT-4 achieves a refined balance between efficient visual-language alignment and the production of high-quality, detailed textual outputs, marking a significant step forward in the field of vision-language understanding.
</details> 

### **MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning**

MiniGPT-v2 acts as a unified interface for vision-language multi-task learning by connecting a static Visual Transformer to a 7B parameter LLaMA-2-chat language model through a linear projection layer, efficiently processing high-resolution images and excelling in various tasks through a three-stage training approach.

[![arXiv](https://img.shields.io/badge/arXiv-2310.09478v3-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.09478v3)  
Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/2354442a-0e96-4010-8b4f-8bc3d666427e" alt="Architecture diagram for MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning" />
</p> 
<details>  
<summary>ℹ️ <i>More Information</i></summary>  
    
**MiniGPT-v2**: A sophisticated model designed to serve as a unified interface for vision-language multi-task learning, leveraging the innovative integration of a visual backbone with a large language model. At its core, the architecture combines a Visual Transformer (ViT) as its visual backbone, which is kept static during training, with **a linear projection layer** that effectively merges every four neighboring visual tokens into one. These consolidated tokens are then projected into the feature space of LLaMA-2-chat, a 7-billion parameter language model, facilitating the processing of high-resolution images (448x448 pixels). This structure allows MiniGPT-v2 to efficiently bridge the gap between visual input and language model processing, catering to a wide array of vision-language tasks. The training methodology employed by MiniGPT-v2 is particularly noteworthy, encompassing a three-stage strategy to comprehensively cover the spectrum of knowledge acquisition and task-specific performance enhancement. Initially, the model is exposed to a mix of weakly-labeled and fine-grained datasets, focusing on broad vision-language understanding. The training progressively shifts towards more fine-grained data to hone in on specific task improvements. In the final stage, MiniGPT-v2 is trained on multi-modal instruction and language datasets, aiming to refine its response to multi-modal instructions. The use of task-specific identifier tokens during training plays a crucial role in reducing ambiguity and sharpening task distinction, enabling the model to adeptly navigate the complexities of vision-language tasks. To support its extensive training and operational capabilities, MiniGPT-v2 utilizes a diverse array of datasets, including LAION, CC3M, SBU, GRIT-20M, COCO caption, and several others, each selected to fulfill distinct stages of the training process—from broad knowledge acquisition to task-specific improvements and sophisticated multi-modal instruction handling. This strategic dataset employment underscores MiniGPT-v2's capacity to assimilate and apply knowledge across a broad range of vision-language contexts, positioning it as a versatile tool in the evolving landscape of multi-task learning interfaces.
</details> 

### **LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents**

LLaVA-Plus pioneers the creation of multimodal agents by integrating diverse vision and vision-language models into a skill repository, enabling the agent to learn and use tools effectively through end-to-end training on comprehensive multimodal instruction-following data.

[![arXiv](https://img.shields.io/badge/arXiv-2311.05437-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.05437) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)  
Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, Chunyuan Li
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/1ede1c4f-bdeb-48e0-ae8e-ccfbee1dea51" alt="Architecture diagram for LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**LLaVA-Plus**: Represents an innovative leap in the design of multimodal agents, integrating a diverse array of vision and vision-language pre-trained models into a comprehensive skill repository. This integration enables LLaVA-Plus to leverage end-to-end training to systematically expand its capabilities, allowing it to activate and combine relevant tools based on the users' multimodal inputs. The architecture of LLaVA-Plus is centered around a unified scheme for representing **multimodal instruction-following data**, which is essential for its advanced end-to-end trained multimodal instruction-following capabilities. The model is distinguished by its training methods, which utilize curated multimodal instruction-following data covering a broad spectrum of tasks, including visual understanding, generation, external knowledge retrieval, and their combinations. This approach allows LLaVA-Plus to incorporate new tools through instruction tuning, thereby expanding its abilities by learning to use these tools effectively. The training datasets—COCO, HierText, InfoSeek, JourneyDB, and Instruct P2P—are meticulously selected to enhance the model's training on visual understanding skills such as detection, segmentation, captioning, OCR, and external knowledge retrieval, alongside generation tasks and skill compositions. LLaVA-Plus employs unique alignment techniques and fusion methods that utilize raw visual signals during human-AI interaction sessions to improve tool use performance, planning, and reasoning. These techniques enable the seamless integration of vision and text embeddings by combining user inputs, tool activation prompts, and execution results into a unified dialogue format. This strategic approach not only facilitates enhanced interaction between the model and its users but also significantly boosts the model's overall performance and versatility in handling complex multimodal tasks.
</details> 

### **BakLLaVA**

BakLLaVA elevates the LLaVA framework by employing a Mistral 7B base enhanced with LLaVA 1.5 architecture, undergoing a meticulous two-stage training process on a diverse dataset to achieve superior performance in multimodal benchmarks, outperforming competitors like Llama 2 13B.

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/skunkworksai/bakllava) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/SkunkworksAI/BakLLaVA-1)
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**BakLLaVA**: Represents an innovative advancement in the realm of AI models, distinguishing itself with significant architectural enhancements over its predecessor, LLaVA. Developed with a strong focus on integrating multimodal capabilities into language models, BakLLaVA leverages a **Mistral 7B** base, augmented with the advanced **LLaVA 1.5 architecture**, to push the boundaries of performance in various benchmarks. This model has been meticulously designed to outperform notable predecessors, such as Llama 2 13B, across several benchmarks, showcasing the efficiency and effectiveness of its underlying architecture .The training methodology of BakLLaVA is particularly noteworthy, employing a feature alignment stage that utilizes 600K filtered CC3M images for establishing a robust vision-language connection. This process is complemented by a visual instruction tuning stage, where 150K GPT-generated multimodal instructions are utilized, signifying a tailored approach towards encoding vision and text together. Such a methodological approach not only enhances feature alignment but also optimizes the model for a broad spectrum of conceptual coverage, efficiency in training, and overall performance. BakLLaVA's architecture benefits from a diverse dataset compilation including 558K filtered image-text pairs from LAION/CC/SBU, captioned by BLIP, alongside 158K GPT-generated multimodal instruction-following data, 450K academic-task-oriented VQA data, and 40K ShareGPT data, among others. This extensive dataset collection is pivotal for the model's training, ensuring broad concept coverage and reinforcing the model's capabilities in feature alignment and visual instruction tuning. The strategic selection of datasets underscores BakLLaVA's commitment to advancing AI's understanding and processing of complex visual and textual information, setting a new standard for multimodal AI models.
</details> 

### **CogVLM: Visual Expert for Pretrained Language Models**

CogVLM enhances pretrained language models with a dedicated visual expert module, incorporating a QKV matrix and MLP within each layer to achieve deep visual-language feature alignment, enabling superior performance in multimodal tasks such as image captioning and visual question answering.

[![arXiv](https://img.shields.io/badge/arXiv-2311.03079v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.03079v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/thudm/cogvlm)  
Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, Jie Tang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/93d951e1-ad49-47fd-9135-c11bc69d49bc" alt="Architecture diagram for CogVLM: Visual Expert for Pretrained Language Models" />
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CogVLM**: This approach enables the model to deeply fuse vision-language features, enhancing its ability to process and understand multimodal inputs. The architecture of CogVLM is built around several key components: a Vision Transformer (ViT) encoder, **an MLP adapter**, a pretrained large language model akin to GPT, and the innovative visual expert module. These components work in tandem to facilitate the model's advanced capabilities in handling complex visual and textual information. The training methodology for CogVLM is comprehensive, encompassing both pretraining and fine-tuning phases. During pretraining, the model undergoes learning with a focus on image captioning loss and Referring Expression Comprehension (REC) across an extensive dataset comprising over 1.5 billion image-text pairs and a visual grounding dataset featuring 40 million images. The fine-tuning phase employs a unified instruction-supervised approach across a variety of visual question-answering datasets, further refining the model's performance. CogVLM's alignment techniques are particularly noteworthy, employing **a visual expert module** in each layer that leverages a **QKV (Query, Key, Value) matrix** and an **MLP (Multilayer Perceptron)** to achieve deep visual-language feature alignment. This method not only allows for the seamless integration of image features into the language model's processing layers but also significantly enhances the model's overall multimodal processing capabilities. The datasets employed in training and refining CogVLM include LAION-2B, COYO-700M, a visual grounding dataset of 40 million images, and several visual question-answering datasets like VQAv2, OKVQA, TextVQA, OCRVQA, and ScienceQA. These datasets serve multiple purposes, from pretraining and instruction alignment to enhancing the model's proficiency in tasks such as image captioning and referring expression comprehension. Through this strategic use of diverse datasets, CogVLM is positioned to excel in a wide array of multimodal tasks, marking a significant advancement in the field of vision-language models.
</details> 

### **CogVLM2: Enhanced Vision-Language Models for Image and Video Understanding**

CogVLM2 is a family of open-source visual language models designed to push the boundaries of image and video understanding. This new generation builds upon the success of previous CogVLM models, focusing on enhanced vision-language fusion, efficient high-resolution architecture, and broader modalities and applications.

[![arXiv](https://img.shields.io/badge/arXiv-2408.16500-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2408.16500) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/THUDM/CogVLM2) [![HuggingFace](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/collections/THUDM/cogvlm2-6645f36a29948b67dc4eef75)  
Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, Lei Zhao, Zhuoyi Yang, Xiaotao Gu, Xiaohan Zhang, Guanyu Feng, Da Yin, Zihan Wang, Ji Qi, Xixuan Song, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Yuxiao Dong, Jie Tang  

<p align="center">
<img src="https://github.com/user-attachments/assets/f60247aa-66b3-486c-891c-c29cefe8aed4" alt="Architecture diagram for CogVLM2: Enhanced Vision-Language Models for Image and Video Understanding" />
</p>

<details>
<summary>ℹ️ <i>More Information</i></summary>

CogVLM2 is a new generation visual language model designed for comprehensive image and video understanding. It leverages a powerful ViT encoder to extract visual features from high-resolution images or video sequences, which are then downsampled by a convolutional layer and aligned with linguistic representations through a SwiGLU module. This adapter efficiently bridges the visual and language modalities while preserving critical image information. The model then utilizes a visual expert architecture, integrating visual features into both the attention and FFN modules of the language decoder. This approach allows for deep vision-language fusion without compromising the model's inherent language capabilities. Notably, CogVLM2-Video extends this architecture to handle videos, incorporating timestamps alongside multi-frame inputs to enable temporal localization and question-answering capabilities. The CogVLM2 family has achieved state-of-the-art results on various benchmarks, including MMBench, MM-Vet, TextVQA, MVBench, and VCG-Bench, showcasing its versatility and effectiveness across a wide range of image and video understanding tasks.
</details>


### **Ferret: Refer and Ground Anything Anywhere at Any Granularity**

FERRET, a multimodal large language model, excels in spatial referencing and grounding by using a hybrid region representation that combines discrete coordinates with continuous features, allowing it to precisely pinpoint objects and regions within images, regardless of their complexity.

[![arXiv](https://img.shields.io/badge/arXiv-2310.07704v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2310.07704v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/apple/ml-ferret)  
Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, Yinfei Yang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/a5ff801f-d523-4383-8b89-e2499976b2bb" alt="Architecture diagram for Ferret: Refer and Ground Anything Anywhere at Any Granularity" />
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**FERRET**: stands as a multimodal large language model (MLLM) that pioneers in spatially referring to any object within an image, irrespective of its shape or granularity, and grounding open-vocabulary descriptions with precision. The architecture of FERRET is distinguished by its hybrid region representation, which marries discrete coordinates with continuous features to depict image regions. This novel approach enables the model to handle a wide range of spatial referring tasks, from pinpointing precise locations to addressing more abstract, shapeless areas within images. At the core of FERRET's architecture are several key components: an image encoder tasked with deriving image embeddings, **a spatial-aware visual sampler** designed to extract regional continuous features, and a language model that integrates image, text, and region features. This intricate setup facilitates the model's unique ability to understand and generate language that refers to spatial elements in images with unprecedented accuracy. The training of FERRET is conducted on the GRIT dataset, which includes over 1.1 million samples imbued with hierarchical spatial knowledge. This process is augmented by spatial-aware visual sampling techniques that cater to the diverse shapes and densities found in spatial data, allowing for the simultaneous generation of text and coordinates for objects within images.FERRET's alignment techniques and fusion methods are particularly noteworthy. By blending discrete coordinates with continuous visual features, the model can process inputs of freely formed regions and ground descriptions in its outputs accurately. This capability is supported by a diverse dataset portfolio, including GRIT for its rich spatial annotations, and Visual Genome, RefCOCOs, and Flickr30k for tasks such as object detection, phrase grounding, and evaluating the model's proficiency in referring and grounding. Through these methodologies, FERRET advances the field of multimodal language models by providing a versatile framework for spatial reasoning and language grounding in visual contexts.
</details> 

### **Fuyu-8B: A Multimodal Architecture for AI Agents**

Fuyu-8B introduces a streamlined architecture for AI agents by directly projecting image patches into a decoder-only transformer, simplifying multimodal processing by treating image and text tokens uniformly, and achieving efficient performance in vision-language tasks despite its straightforward design.

[![Link](https://img.shields.io/badge/https%3A%2F%2Fwww.adept.ai%2Fblog%2Ffuyu-8b?style=flat&label=Fuyu%208B
)](https://www.adept.ai/blog/fuyu-8b) [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/adept/fuyu-8b)  
Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar

<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/61a75fb4-ced7-419c-bff7-7cb2e3ddc02d" alt="Architecture diagram for Fuyu-8B: A Multimodal Architecture for AI Agents" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**Fuyu-8B**: A streamlined multimodal model tailored for digital agents, distinguished by its unique approach to handling visual data and its integration with textual information. At the core of Fuyu-8B's architecture is a decoder-only transformer, a departure from traditional models that rely on separate image encoders. This design facilitates the direct projection of image patches into the transformer's initial layer with **a linear projection**, allowing Fuyu-8B to process images of any resolution without the need for complex training stages or the integration of resolution-specific mechanisms. The simplicity of this architecture does not only lie in its unified processing of image and text data but also in its elimination of the need for cross-attention mechanisms or adapters, streamlining the model's training and inference processes. In terms of alignment techniques, Fuyu-8B employs a novel approach by treating image tokens on par with text tokens from the inception of the model's processing pipeline. This method does away with separate position embeddings for images, thereby simplifying the alignment process between textual and visual data. The model's ability to support arbitrary image resolutions and perform fine-grained localization is particularly advantageous for applications requiring detailed visual understanding alongside textual interaction. The datasets utilized in Fuyu-8B's development, including VQAv2, OKVQA, COCO Captions, and AI2D, are instrumental in benchmarking the model against standard image understanding tasks such as visual question answering and caption generation. Despite Fuyu-8B's primary focus on applications within digital agents, the selection of these datasets ensures a comprehensive evaluation of its capabilities in broader contexts of image understanding and multimodal interaction. Through its innovative architecture and methodological simplicity, Fuyu-8B sets a new direction for the development of AI agents capable of sophisticated multimodal reasoning.
</details>

### **OtterHD: A High-Resolution Multi-modality Model**

OtterHD-8B, inspired by Fuyu-8B, directly integrates pixel-level information from high-resolution images (up to 1024x1024 pixels) into its language model using position embeddings, eliminating the need for a separate vision encoder and enabling precise interpretation of detailed visual inputs alongside textual instructions.

[![arXiv](https://img.shields.io/badge/arXiv-2311.04219v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.04219v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/luodian/otter)

Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, Ziwei Liu
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**OtterHD-8B**: Represents an evolutionary step in multi-modality model design, building on the foundation of the **Fuyu-8B architecture** to interpret high-resolution visual inputs with exceptional precision. Unlike traditional models limited by fixed-size vision encoders, OtterHD-8B is equipped to handle flexible input dimensions, allowing for enhanced versatility across a variety of inference requirements. This model integrates pixel-level visual information directly into the language model without the need for a separate vision encoder, employing position embeddings to comprehend varying image sizes and enabling the processing of high-resolution images up to 1024x1024 pixels. Instruction tuning in OtterHD-8B is tailored towards accommodating various image resolutions, with the model being trained on a diverse dataset mixture including LLaVA-Instruct, VQAv2, GQA, OKVQA, OCRVQA, A-OKVQA, COCO-GOI, COCO-Caption, TextQA, RefCOCO, COCO-ITM, ImageNet, and LLaVA-RLHF. This training employs FlashAttention-2 and other fused operators for optimization, leveraging PyTorch and Hugging Face Transformers. The direct integration of pixel-level information into the language model, facilitated by position embeddings, enables OtterHD-8B to understand and generate responses to high-resolution images alongside textual instructions without conventional vision and text embedding fusion methods. The datasets chosen for training OtterHD-8B underscore its focus on a broad array of vision and language tasks, including question answering, object recognition, and text-image alignment, aiming to enhance the model's capabilities in these areas. By directly processing image patches alongside textual instructions, OtterHD-8B eschews traditional fusion methods, leveraging its architecture to interpret and respond to complex multimodal inputs. This approach not only marks a significant advancement in handling high-resolution images but also in the model's overall ability to comprehend and interact with visual and textual data, positioning OtterHD-8B as a notable development in the field of multi-modality models.
</details>

### **SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models**

SPHINX pushes the boundaries of multi-modal LLMs by jointly mixing model weights, tasks, and visual embeddings during training, utilizing a two-stage approach that unfreezes the LLM (LLaMA-2) during pre-training for enhanced cross-modal learning and achieving impressive performance on a variety of vision-language tasks.

[![arXiv](https://img.shields.io/badge/arXiv-2311.07575v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2311.07575v1) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/alpha-vllm/)
Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, Yu Qiao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/3a1bf3fa-d0c5-4692-b9a8-97bea41ce226" alt="Architecture diagram for SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models" />
</p> 
<details>
<summary>ℹ️ <i>More Information</i></summary>  
    
**SPHINX**: stands out as a multi-modal large language model (MLLM) designed to enhance the integration of language and vision through an innovative approach that includes the **joint mixing of model weights**, tuning tasks, and visual embeddings. This model is particularly distinguished by its methodology of unfreezing the large language model during pre-training to foster more effective cross-modal learning. The architecture of SPHINX is built upon a foundation that combines vision encoders, **two linear projection layers**, and leverages LLaMA-2 as the language model backbone. It adopts a two-stage training paradigm that emphasizes pre-training for vision-language alignment followed by fine-tuning aimed at visual instruction-following tasks. In the realm of training methodologies, SPHINX employs a strategy that emphasizes **the joint mixing of model weights**, tuning tasks, and visual embeddings, setting a precedent for robust cross-modal knowledge acquisition. This approach is complemented by a pre-training regimen that utilizes both real-world and synthetic data, thereby ensuring a comprehensive understanding across various visual instruction tasks. The model introduces an efficient strategy for processing high-resolution images, utilizing mixed scales and sub-images to accommodate diverse visual inputs. Moreover, SPHINX achieves vision-language alignment by integrating comprehensive visual embeddings, unfreezing the LLM during pre-training, and employing a weight-mixing strategy that bridges domain-specific knowledge across different network architectures and training paradigms. The datasets utilized in training SPHINX, including LAION-400M, LAION-COCO, RefinedWeb, VQAV2, GQA, OKVQA, A-OKVQA, OCRVQA, TextCaps, COCO, LVIS, RefCOCO, VG, and Flickr30k, serve a multifaceted purpose. They are instrumental in achieving multi-modal alignment, language-only tuning, and addressing a wide spectrum of visual question answering and general vision tasks. These tasks range from object detection and human pose estimation to referring object localization and understanding descriptions within the context of image regions. SPHINX, through its meticulous design and strategic training approach, sets a new benchmark in the field of multi-modal large language models, advancing the capabilities in vision-language integration.
</details>

### **CLIP: Contrastive Language-Image Pre-training**

CLIP leverages a contrastive learning approach, training separate image and text encoders on a massive dataset of 400 million image-text pairs to predict the most relevant captions for images, enabling impressive zero-shot transfer capabilities to various downstream tasks without requiring task-specific training data.

[![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2103.00020) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/openai/CLIP)  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/c335c342-9a2c-4d4e-83d6-d3077cc32643" alt="Architecture diagram for CLIP: Contrastive Language-Image Pre-training" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**CLIP**: model represents a groundbreaking approach in the field of machine learning, aiming to bridge the gap between visual and textual information through natural language supervision. Its architecture is designed to understand and predict **the most fitting captions for given images**, a methodology that stems from its training on a vast dataset of 400 million image-text pairs. This extensive training enables CLIP to learn state-of-the-art (SOTA) image representations and apply this knowledge to a wide range of downstream tasks without the need for task-specific training data, facilitating zero-shot transfer capabilities. At the core of CLIP are two primary components: **an image encoder** and **a text encoder**. These encoders are trained using a contrastive learning approach, optimizing for a contrastive objective that seeks to maximize the cosine similarity between correct image-text pairs while minimizing it for incorrect ones. This process is achieved through **a symmetric cross-entropy loss over the similarity scores between the embeddings of images and texts**, enabling the model to effectively link visual concepts with their linguistic descriptions. The model's ability to generalize across various tasks is further enhanced by its training methodology and the specific datasets it utilizes. By covering a broad spectrum of visual concepts and leveraging natural language for supervision, CLIP is adept at learning representations that are highly transferable to new tasks and domains. The custom dataset of 400 million image-text pairs, curated from the internet, plays a pivotal role in this process, providing the diverse and extensive visual and textual information necessary for the model to learn effectively. Through these innovations, CLIP sets a new standard for learning transferable visual models, showcasing the power of natural language in facilitating robust and versatile visual understanding.
</details> 

### **MetaCLIP: Demystifying CLIP Data**

MetaCLIP refines the data curation process for training vision-language models by employing algorithms that leverage CLIP-derived metadata to create a balanced and high-quality dataset from vast sources like CommonCrawl, resulting in improved performance and diversity compared to models trained on CLIP's original dataset.

[![arXiv](https://img.shields.io/badge/arXiv-2309.16671-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2309.16671) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/MetaCLIP)  
Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, Christoph Feichtenhofer
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/a6c79d0e-a4c7-48c9-86b6-3a8cc9853e11" alt="Architecture diagram for MetaCLIP: Demystifying CLIP Data" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**MetaCLIP**: Represents an innovative approach in the realm of data curation for machine learning, specifically targeting the **enhancement of training datasets** through metadata utilization derived from CLIP's concepts. This model is designed to sift through extensive raw data pools, such as the CommonCrawl dataset, to curate a high-quality, balanced subset that significantly betters the diversity and performance metrics of the data used for training machine learning models. The essence of MetaCLIP lies in its unique architecture that incorporates data curation algorithms, which are adept at leveraging metadata for the purpose of balancing and enriching the training dataset both in terms of quality and diversity. The architecture of MetaCLIP is structured around these **data curation algorithms**, which play a pivotal role in the framework by identifying and assembling a balanced and high-quality dataset from a vast collection of 400 million image-text pairs initially sourced from CommonCrawl. This process is instrumental in MetaCLIP's ability to demonstrate superior performance on various benchmarks, including zero-shot ImageNet classification, when compared to datasets curated using CLIP's original methodologies. The training methods employed by MetaCLIP, therefore, are not just about processing and learning from data but also about intelligently selecting the data that is most beneficial for the training process, ensuring that the model is trained on a dataset that is representative, diverse, and of high quality. The purpose behind employing datasets like CommonCrawl within the MetaCLIP framework is to address and overcome the limitations observed in CLIP's original dataset. By curating a balanced and high-quality dataset of 400 million image-text pairs, MetaCLIP sets a new precedent in the field of machine learning data curation. This strategic selection and enhancement of the training dataset enable MetaCLIP to significantly improve performance on standard benchmarks compared to its predecessor, highlighting the importance of dataset quality and diversity in achieving high performance in machine learning tasks. Through its innovative approach to data curation, MetaCLIP offers a promising avenue for enhancing the capabilities of machine learning models, particularly in applications requiring robust image-text understanding and classification.
</details> 

### **Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**

Alpha-CLIP builds upon the CLIP model by incorporating region awareness through the addition of an alpha channel to the image encoder, trained on millions of RGBA region-text pairs, enabling precise control over image emphasis and enhancing performance across various tasks requiring detailed spatial understanding.

[![arXiv](https://img.shields.io/badge/arXiv-2312.03818-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.03818) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/SunzeY/AlphaCLIP)

Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, Jiaqi Wang
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/07bd6161-1682-4954-97f3-3770258bfa8c" alt="Architecture diagram for Alpha-CLIP: A CLIP Model Focusing on Wherever You Want" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
  
**Alpha-CLIP**: Introduces a significant enhancement to the original CLIP model, incorporating region awareness to its repertoire of capabilities. This model is fine-tuned on millions of RGBA region-text pairs, enabling it to maintain CLIP's visual recognition prowess while offering precise control over the emphasis of image content. By integrating an additional **alpha channel into the CLIP image encoder**, Alpha-CLIP allows for detailed segmentation and region-specific processing without modifying the foundational CLIP weights, thus facilitating a nuanced approach to image understanding that respects the spatial dynamics of visual data.
The training of Alpha-CLIP leverages a novel data generation pipeline designed to produce a vast array of RGBA-region text pairs. This process involves the creation of natural images equipped with foreground alpha channels and their corresponding referring expressions for specific regions. Such a methodology not only enables the fine-tuning of the model with an additional alpha channel input but also underpins its ability to perform with heightened specificity across various tasks. These tasks range from image recognition to multimodal large language models, and extend into both 2D and 3D generation domains, showcasing Alpha-CLIP's versatility and broad applicability. Datasets like LAION-400M, LAION-5B, and GRIT play a crucial role in training Alpha-CLIP, providing a wide spectrum of images for initial training and fine-grained mask-level labels for enhancing local perception capabilities. This strategic choice of datasets ensures that Alpha-CLIP is not only well-equipped for general visual recognition tasks but also capable of nuanced, region-specific processing and understanding, setting a new standard for models at the intersection of language and vision.
</details> 

### **GLIP: Grounded Language-Image Pre-training**

GLIP revolutionizes language-image pre-training by unifying object detection and phrase grounding, allowing it to understand and execute tasks requiring object-level precision and language awareness through a deep integration of visual and textual information during training.

[![arXiv](https://img.shields.io/badge/arXiv-2112.03857-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2112.03857) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/microsoft/GLIP)  
Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/06e6f8dc-fbd8-49da-8651-a22ee2edcf3d" alt="Architecture diagram for GLIP: Grounded Language-Image Pre-training" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**GLIP**: A novel approach that innovatively unifies the tasks of object detection and phrase grounding by redefining object detection as a phrase grounding challenge. This strategic reformation allows the model to exploit extensive image-text paired datasets for pre-training, equipping it with the capability to comprehend and execute tasks that require object-level precision, language awareness, and semantically rich visual representations. At its core, GLIP's architecture is designed to deeply integrate visual and textual information, enhancing its understanding of complex visual scenes in conjunction with textual prompts. The architecture of GLIP is composed of several critical components, including a visual encoder that can either be a Convolutional Neural Network (CNN) or a Transformer, tasked with extracting features from regions or bounding boxes within images. It also includes a language encoder dedicated to processing text prompts and prediction heads (box classifier and box regressor) that are trained using **classification** and **localization loss**. A distinctive feature of GLIP is its method of deep fusion between image and text, specifically in the latter stages of encoding, which merges visual and textual information more comprehensively than traditional methods. GLIP's training methodology is as innovative as its architecture, employing a unified formulation that amalgamates detection and grounding tasks into a singular workflow. This model is trained end-to-end, optimizing losses defined for **both detection** (focusing on localization and classification) and **grounding** (centering on alignment scores between image regions and corresponding words in the prompt). Such deep integration of visual and language features during training is pivotal, facilitating the model's ability to learn effectively from paired image-text data. The datasets utilized for training GLIP, including COCO, OpenImages, Objects365, Visual Genome, Flickr30k-entities, LVIS, and PhraseCut, are meticulously selected to cover a wide array of object classes and scenarios, each serving a unique purpose from object detection and phrase grounding to instance segmentation and referring expression segmentation. Through this comprehensive training, GLIP sets a new precedent in the realm of language-image pre-training, demonstrating advanced capabilities in interpreting and interacting with both visual and textual data.
</details>

### **ImageBind: One Embedding Space To Bind Them All**

ImageBind revolutionizes multimodal learning by creating a single, joint embedding space that integrates six modalities – images, text, audio, depth, thermal, and IMU data – through image-paired data as a central binding agent, allowing for zero-shot classification and retrieval across diverse data types.

[![arXiv](https://img.shields.io/badge/arXiv-2305.05665-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.05665) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/facebookresearch/imagebind)  
Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/fbf8bcdd-b1bb-4fd8-8723-3c82e84ef759" alt="Architecture diagram for ImageBind: One Embedding Space To Bind Them All" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**ImageBind**: Introduces an innovative approach to multimodal learning by creating **a joint embedding space** that encompasses six different modalities: **images, text, audio, depth, thermal, and IMU (Inertial Measurement Unit)** data. This model uniquely employs image-paired data as a central binding agent, enabling it to leverage the capabilities of large-scale vision-language models to extend zero-shot capabilities to new, previously unlinked modalities. By doing so, ImageBind not only facilitates a deeper integration of diverse data types but also opens up new avenues for zero-shot classification and retrieval across a wide range of applications. At the heart of ImageBind's architecture lies a transformer-based design, adapted for each specific modality to ensure optimal processing and representation. For instance, it utilizes a Vision Transformer for image data, with each modality encoder being augmented by **modality-specific linear projection heads**. These adaptations are crucial for maintaining a uniform embedding size across the disparate data types, ensuring that the model can effectively learn from and link together the various modalities. This uniformity is key to ImageBind's ability to create a cohesive and comprehensive embedding space that captures the nuances of each data type. The training methodology behind ImageBind is particularly noteworthy. It employs contrastive learning, utilizing both web-scale image-text data and naturally occurring paired data from various modalities, such as video-audio and image-depth pairs. This strategy allows the model to learn a single joint embedding space without requiring all modalities to co-occur, a significant advantage that enhances its flexibility and applicability. The use of datasets like Audioset, SUN RGB-D, LLVIP, and Ego4D, which provide naturally paired data across the model's target modalities, is critical to this process. These datasets enable ImageBind to achieve emergent zero-shot classification and retrieval performance on tasks tailored to each modality, showcasing the model's ability to seamlessly navigate and leverage the complex interplay between different forms of data.
</details>

### **SigLIP: Sigmoid Loss for Language Image Pre-Training**

SigLIP introduces a simple pairwise sigmoid loss for language-image pre-training, allowing for scalable training with large batch sizes without compromising performance, enabling efficient alignment between image and text representations.

[![arXiv](https://img.shields.io/badge/arXiv-2303.15343-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.15343)  
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer  
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/60018313-37dd-4dbd-8eb4-a3075fd26663" alt="Architecture diagram for SigLIP: Sigmoid Loss for Language Image Pre-Training" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**SigLIP**: A novel approach to language-image pre-training by proposing **a simple pairwise sigmoid loss**. This method contrasts with standard contrastive learning that utilizes softmax normalization, as it operates directly on image-text pairs without necessitating a global view of pairwise similarities for normalization. The primary advantage of this approach is its scalability, allowing for the use of larger batch sizes without compromising performance. The architecture leverages a vision transformer for image processing and a conventional transformer for text, with the sigmoid loss facilitating independent processing of image-text pairs. This design enables more efficient training dynamics, particularly in the context of large batch sizes, by examining the effects of varying the negative to positive ratio and the selection of example pairs. Training methodologies focus on exploiting large batch sizes, delving into the dynamics of how batch size variations influence model performance. The introduction of sigmoid loss is pivotal, enabling the model to train effectively with these large batches by investigating the relationship between the ratio of negative to positive examples and the optimization of example pair selection. The use of the LiT image-text dataset and the WebLI dataset is integral to the model's training, aiming to achieve aligned representational spaces between images and texts. These datasets are chosen for their utility in assessing zero-shot transfer capabilities, as well as in exploring the scalability and efficiency of the model's sigmoid loss-based training. In essence, SigLIP marks a significant stride in language-image pre-training through its innovative use of sigmoid loss, enhancing scalability and training efficiency. This approach not only simplifies the training process by eliminating the need for global normalization but also showcases the model's adaptability to large-scale data handling. The strategic selection of datasets further underscores the model's capability to forge aligned representational spaces, paving the way for advanced zero-shot learning and efficient multimodal integration.
</details>

### **ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

The Vision Transformer (ViT) revolutionizes image recognition by applying the Transformer architecture to images, processing them as a sequence of fixed-size patches, thereby demonstrating that image recognition can benefit from the power of transformers, surpassing traditional convolutional neural network (CNN) approaches with the aid of large-scale training datasets.

[![arXiv](https://img.shields.io/badge/arXiv-2010.11929v2-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2010.11929v2) [![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/google-research/vision_transformer)  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
<p align="center">
<img src="https://github.com/gokayfem/Awesome-VLM-Architectures/assets/88277926/b2f77966-c2e8-4204-ba90-be51196a7dee" alt="Architecture diagram for ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" />
</p> 
<details> 
<summary>ℹ️ <i>More Information</i></summary>  
    
**The Vision Transformer (ViT)**: A paradigm shift in image recognition by applying the transformer architecture, predominantly used in natural language processing, directly to images. It innovatively processes images as **a sequence of fixed-size patches**, akin to how tokens are treated in **text applications**. This approach is facilitated through minimal modifications to the standard transformer components, emphasizing the model's adaptability to visual tasks without relying on the convolutional neural networks' (CNNs) inductive biases. ViT's architecture is distinguished by its use of linear embedding for **image patches** and **position embeddings**, which are crucial for maintaining the spatial hierarchy of image data. The core of ViT is a standard Transformer encoder that includes multiheaded self-attention (MSA) and multilayer perceptron (MLP) blocks, complemented by layer normalization and residual connections, underscoring its efficiency and robustness in handling visual data. Training methodologies for ViT are characterized by its scalability and the significant impact of dataset size on its performance. Initially, ViT exhibits modest accuracies without strong regularization techniques. However, its performance escalates with the scale of training, showcasing its potential to outperform traditional CNN approaches through extensive pre-training on large datasets. This process highlights the critical role of dataset selection in ViT's training regimen. It is fine-tuned on smaller datasets following a comprehensive pre-training phase that leverages large datasets like ImageNet-21k and JFT-300M to enhance model generalization and performance across a wide range of tasks. The datasets employed, including ImageNet, CIFAR-100, VTAB, ImageNet-21k, and JFT-300M, serve dual purposes: benchmarking the model's image classification capabilities and evaluating its transferability to diverse tasks with limited data, thereby establishing ViT's versatility and effectiveness in advancing image recognition tasks.
</details>

## Important References

- [Guide to Vision-Language Models (VLMs) by Görkem Polat](https://encord.com/blog/vision-language-models-guide/)
- [VLM Primer by Aman Chadha](https://aman.ai/primers/ai/VLM/#google_vignette)
- [Generalized Visual Language Models by Lilian Weng](https://lilianweng.github.io/posts/2022-06-09-vlm/)
