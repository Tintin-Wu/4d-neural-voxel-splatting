# Project Page Design — 4D Neural Voxel Splatting

**Date:** 2026-05-10  
**Venue:** IEEE ICIP 2026  
**Hosting:** GitHub Pages on this repo (`https://<username>.github.io/4d-neural-voxel-splatting/`)

---

## Overview

A single-file static HTML project page served via GitHub Pages. No build tools, no frameworks, no external dependencies — one `index.html` at the repo root that GitHub Pages picks up automatically. Style is Academic Clean: white background, Georgia serif typography (system font), pill-shaped link buttons, inspired by the Nerfies / NeRF project page aesthetic.

---

## Authors & Affiliations

| Name | Affiliation |
|---|---|
| Chun Tin Wu | ¹ National Taiwan University |
| Jun Cheng Chen | ² Academia Sinica |

---

## Links

| Button | Target | Status |
|---|---|---|
| Paper (ICIP) | TBD | Grayed-out, labeled "Coming Soon" |
| arXiv | https://arxiv.org/abs/2511.00560 | Live |
| Code | This repo (GitHub) | Live — links to repo root |

---

## Page Sections (top → bottom)

### 1. Header
- Venue tag: `IEEE ICIP 2026` (small caps, muted)
- Title: `4D Neural Voxel Splatting` (large, bold serif)
- Authors with superscript affiliation numbers
- Affiliation line below authors
- Link buttons: Paper (Coming Soon) · arXiv · Code

### 2. Teaser Video
- `assets/Appendix_1.mp4` embedded as `<video autoplay muted loop playsinline>`
- Full-width, light gray background band
- No caption needed

### 3. Abstract
- Section heading: `Abstract` (small caps)
- Full abstract text, justified, serif
- Full text:
  > Although 3D Gaussian Splatting (3D-GS) achieves efficient rendering for novel view synthesis, extending it to dynamic scenes still results in substantial memory overhead from replicating Gaussians across frames. To address this challenge, we propose 4D Neural Voxel Splatting (4D-NVS), which combines voxel-based representations with neural Gaussian splatting for efficient dynamic scene modeling. Instead of generating separate Gaussian sets per timestamp, our method employs a compact set of neural voxels with learned deformation fields to model temporal dynamics. The design greatly reduces memory consumption and accelerates training while preserving high image quality. We further introduce a novel view refinement stage that selectively improves challenging viewpoints through targeted optimization, maintaining global efficiency while enhancing rendering quality for difficult viewing angles. Experiments demonstrate that our method outperforms state-of-the-art approaches with significant memory reduction and faster training, enabling real-time rendering with superior visual fidelity.

### 4. Method Overview
- Section heading: `Method Overview`
- `assets/overview.png` full-width image
- Figure caption: "Overview of 4D-NVS. Anchor neural voxels are decoded into Gaussian attributes, which are deformed through a multi-resolution HexPlane field and rendered via differentiable Gaussian rasterization."

### 5. Supplementary Results
- Section heading: `Supplementary Results`
- 2-column video grid:
  - Row 1: Appendix_2.mp4 | Appendix_3.mp4
  - Row 2: Appendix_4.mp4 — full width, spanning both columns
- All videos: autoplay, muted, loop, playsinline

### 6. BibTeX
- Section heading: `BibTeX`
- Monospace code block with copy-on-click behavior
- Citation:
  ```bibtex
  @inproceedings{wu2026_4dnvs,
    title     = {4D Neural Voxel Splatting},
    author    = {Wu, Chun Tin and Chen, Jun Cheng},
    booktitle = {IEEE International Conference on Image Processing (ICIP)},
    year      = {2026}
  }
  ```

### 7. Footer / Acknowledgments
- One short paragraph acknowledging prior work: 3D-GS, Scaffold-GS, 4DGaussians, HexPlane
- "Website template inspired by Nerfies" with link

---

## Implementation Notes

- **File**: `index.html` at repo root (GitHub Pages serves this automatically)
- **CSS**: Inline `<style>` block — no external stylesheet, no CDN
- **Fonts**: System fonts — `Georgia, serif` for body text; `system-ui, sans-serif` for UI elements (buttons, labels, code)
- **Responsive**: Single-column layout, max-width 860px, centered — readable on mobile without a grid system
- **Video fallback**: `<source>` tag with MP4; graceful degradation if video unsupported
- **BibTeX copy**: Small `<button>` that copies inner text to clipboard via `navigator.clipboard`
- **GitHub Pages config**: No `_config.yml` needed — Pages auto-serves `index.html` from `main` branch root

---

## Out of Scope

- Carousel / interactive viewer
- Comparison sliders
- Dark mode toggle
- Results table (no quantitative numbers provided)
