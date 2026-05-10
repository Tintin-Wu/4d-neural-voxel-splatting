# Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Academic Clean HTML project page for "4D Neural Voxel Splatting" (ICIP 2026), served via GitHub Pages.

**Architecture:** One `index.html` at the repo root. All CSS is in an inline `<style>` block. All JS is inline `<script>` at the bottom. No external dependencies, no build step — push to `main` and GitHub Pages serves it immediately.

**Tech Stack:** Plain HTML5, CSS3, vanilla JS (`navigator.clipboard`). No frameworks, no CDNs.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `index.html` | **Create** | Entire project page — shell, CSS, all sections |

No other files are created or modified. All assets referenced are already present in `assets/`.

---

### Task 1: HTML shell, CSS, and header section

**Files:**
- Create: `index.html`

- [ ] **Step 1: Create `index.html` with the full CSS and header section**

Create `index.html` at the repo root with this exact content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>4D Neural Voxel Splatting</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: Georgia, serif;
      background: #fff;
      color: #222;
      line-height: 1.65;
    }

    .container {
      max-width: 860px;
      margin: 0 auto;
      padding: 0 24px;
    }

    /* ── Header ── */
    header {
      text-align: center;
      padding: 52px 24px 36px;
    }
    .venue {
      font-family: system-ui, sans-serif;
      font-size: 12px;
      color: #999;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 14px;
    }
    h1 {
      font-size: 2.1rem;
      font-weight: 700;
      line-height: 1.25;
      margin-bottom: 18px;
      color: #111;
    }
    .authors {
      font-family: system-ui, sans-serif;
      font-size: 15px;
      color: #444;
      margin-bottom: 6px;
    }
    .authors sup { font-size: 10px; vertical-align: super; }
    .affiliations {
      font-family: system-ui, sans-serif;
      font-size: 13px;
      color: #888;
      margin-bottom: 26px;
    }
    .links {
      display: flex;
      justify-content: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn {
      display: inline-block;
      font-family: system-ui, sans-serif;
      font-size: 14px;
      padding: 6px 20px;
      border-radius: 20px;
      border: 1px solid #ccc;
      background: #f4f4f4;
      color: #333;
      text-decoration: none;
      transition: background 0.15s;
    }
    .btn:hover { background: #e8e8e8; }
    .btn.disabled {
      color: #bbb;
      border-color: #e4e4e4;
      cursor: not-allowed;
      pointer-events: none;
    }

    /* ── Section chrome ── */
    section { padding: 48px 0; border-top: 1px solid #eee; }
    .section-heading {
      font-family: system-ui, sans-serif;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      text-align: center;
      color: #777;
      margin-bottom: 28px;
    }

    /* ── Teaser band ── */
    .teaser-band {
      background: #f5f5f5;
      border-top: 1px solid #eee;
      border-bottom: 1px solid #eee;
      padding: 28px 0;
    }
    .teaser-band video {
      width: 100%;
      max-width: 860px;
      display: block;
      margin: 0 auto;
      border-radius: 4px;
    }

    /* ── Abstract ── */
    .abstract-text {
      font-size: 16px;
      line-height: 1.85;
      text-align: justify;
      color: #333;
    }

    /* ── Method figure ── */
    .method-figure { width: 100%; border-radius: 4px; display: block; }
    .figure-caption {
      font-family: system-ui, sans-serif;
      font-size: 13px;
      color: #666;
      text-align: center;
      margin-top: 14px;
      line-height: 1.55;
    }

    /* ── Video grid ── */
    .video-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .video-grid video { width: 100%; border-radius: 4px; display: block; }
    .video-full { margin-top: 16px; }
    .video-full video { width: 100%; border-radius: 4px; display: block; }

    /* ── BibTeX ── */
    .bibtex-wrap { position: relative; }
    .bibtex-block {
      background: #f8f8f8;
      border: 1px solid #e0e0e0;
      border-radius: 6px;
      padding: 20px 20px 20px 20px;
      font-family: 'Courier New', Courier, monospace;
      font-size: 13px;
      line-height: 1.75;
      color: #333;
      overflow-x: auto;
      white-space: pre;
    }
    .copy-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      font-family: system-ui, sans-serif;
      font-size: 12px;
      padding: 4px 14px;
      border-radius: 4px;
      border: 1px solid #ccc;
      background: #fff;
      cursor: pointer;
      color: #555;
      transition: background 0.15s;
    }
    .copy-btn:hover { background: #f0f0f0; }

    /* ── Footer ── */
    footer {
      padding: 36px 24px;
      text-align: center;
      border-top: 1px solid #eee;
      font-family: system-ui, sans-serif;
      font-size: 13px;
      color: #999;
      line-height: 1.8;
    }
    footer a { color: #777; text-decoration: none; }
    footer a:hover { text-decoration: underline; }

    /* ── Responsive ── */
    @media (max-width: 600px) {
      h1 { font-size: 1.55rem; }
      .video-grid { grid-template-columns: 1fr; }
      .video-full { margin-top: 0; }
    }
  </style>
</head>
<body>

  <!-- ── 1. HEADER ── -->
  <header>
    <p class="venue">IEEE International Conference on Image Processing (ICIP) 2026</p>
    <h1>4D Neural Voxel Splatting</h1>
    <p class="authors">
      Chun Tin Wu<sup>1</sup>&ensp;
      Jun Cheng Chen<sup>2</sup>
    </p>
    <p class="affiliations">
      <sup>1</sup>&thinsp;National Taiwan University &emsp;
      <sup>2</sup>&thinsp;Academia Sinica
    </p>
    <div class="links">
      <span class="btn disabled">📄 Paper (Coming Soon)</span>
      <a class="btn" href="https://arxiv.org/abs/2511.00560" target="_blank" rel="noopener">📝 arXiv</a>
      <a class="btn" href="https://github.com/Tintin-Wu/4d-neural-voxel-splatting" target="_blank" rel="noopener">💻 Code</a>
    </div>
  </header>

  <!-- remaining sections go here -->

  <script>
    function copyBibtex() {
      const text = document.getElementById('bibtex-text').innerText;
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copy-btn');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      });
    }
  </script>

</body>
</html>
```

- [ ] **Step 2: Open in browser and verify header**

```bash
# Open index.html directly in browser (no server needed for this step)
xdg-open index.html   # Linux
# or: open index.html  # macOS
```

Expected: Title centered, authors with superscripts, three buttons — "Paper (Coming Soon)" grayed-out, arXiv and Code as active links.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add project page header section"
```

---

### Task 2: Teaser video section

**Files:**
- Modify: `index.html` — insert teaser band after `</header>`

- [ ] **Step 1: Insert teaser band**

Replace the comment `<!-- remaining sections go here -->` in `index.html` with:

```html
  <!-- ── 2. TEASER VIDEO ── -->
  <div class="teaser-band">
    <video autoplay muted loop playsinline>
      <source src="assets/Appendix_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

  <!-- remaining sections go here -->
```

- [ ] **Step 2: Verify in browser**

Reload `index.html`. Expected: full-width video plays silently in a light gray band directly below the header.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add teaser video section"
```

---

### Task 3: Abstract section

**Files:**
- Modify: `index.html` — insert abstract section after teaser band

- [ ] **Step 1: Insert abstract section**

Replace `<!-- remaining sections go here -->` with:

```html
  <!-- ── 3. ABSTRACT ── -->
  <section>
    <div class="container">
      <p class="section-heading">Abstract</p>
      <p class="abstract-text">
        Although 3D Gaussian Splatting (3D-GS) achieves efficient rendering for
        novel view synthesis, extending it to dynamic scenes still results in
        substantial memory overhead from replicating Gaussians across frames. To
        address this challenge, we propose <strong>4D Neural Voxel Splatting
        (4D-NVS)</strong>, which combines voxel-based representations with neural
        Gaussian splatting for efficient dynamic scene modeling. Instead of
        generating separate Gaussian sets per timestamp, our method employs a
        compact set of neural voxels with learned deformation fields to model
        temporal dynamics. The design greatly reduces memory consumption and
        accelerates training while preserving high image quality. We further
        introduce a novel view refinement stage that selectively improves
        challenging viewpoints through targeted optimization, maintaining global
        efficiency while enhancing rendering quality for difficult viewing angles.
        Experiments demonstrate that our method outperforms state-of-the-art
        approaches with significant memory reduction and faster training, enabling
        real-time rendering with superior visual fidelity.
      </p>
    </div>
  </section>

  <!-- remaining sections go here -->
```

- [ ] **Step 2: Verify in browser**

Reload `index.html`. Expected: "ABSTRACT" heading in small caps above justified serif paragraph.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add abstract section"
```

---

### Task 4: Method Overview section

**Files:**
- Modify: `index.html` — insert method section after abstract

- [ ] **Step 1: Insert method overview section**

Replace `<!-- remaining sections go here -->` with:

```html
  <!-- ── 4. METHOD OVERVIEW ── -->
  <section>
    <div class="container">
      <p class="section-heading">Method Overview</p>
      <img class="method-figure" src="assets/overview.png"
           alt="4D-NVS method overview diagram">
      <p class="figure-caption">
        Overview of 4D-NVS. Anchor neural voxels are decoded into Gaussian
        attributes, which are deformed through a multi-resolution HexPlane field
        and rendered via differentiable Gaussian rasterization.
      </p>
    </div>
  </section>

  <!-- remaining sections go here -->
```

- [ ] **Step 2: Verify in browser**

Reload `index.html`. Expected: overview.png displayed full-width with caption below in small system-ui font.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add method overview section"
```

---

### Task 5: Supplementary Results video grid

**Files:**
- Modify: `index.html` — insert results section after method overview

- [ ] **Step 1: Insert supplementary results section**

Replace `<!-- remaining sections go here -->` with:

```html
  <!-- ── 5. SUPPLEMENTARY RESULTS ── -->
  <section>
    <div class="container">
      <p class="section-heading">Supplementary Results</p>
      <div class="video-grid">
        <video autoplay muted loop playsinline>
          <source src="assets/Appendix_2.mp4" type="video/mp4">
        </video>
        <video autoplay muted loop playsinline>
          <source src="assets/Appendix_3.mp4" type="video/mp4">
        </video>
      </div>
      <div class="video-full">
        <video autoplay muted loop playsinline>
          <source src="assets/Appendix_4.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </section>

  <!-- remaining sections go here -->
```

- [ ] **Step 2: Verify in browser**

Reload `index.html`. Expected: two videos side-by-side in the top row, one full-width video below. On a viewport narrower than 600px, all three stack to single column.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add supplementary results video grid"
```

---

### Task 6: BibTeX section

**Files:**
- Modify: `index.html` — insert BibTeX section after supplementary results

- [ ] **Step 1: Insert BibTeX section**

Replace `<!-- remaining sections go here -->` with:

```html
  <!-- ── 6. BIBTEX ── -->
  <section>
    <div class="container">
      <p class="section-heading">BibTeX</p>
      <div class="bibtex-wrap">
        <button class="copy-btn" id="copy-btn" onclick="copyBibtex()">Copy</button>
        <div class="bibtex-block" id="bibtex-text">@inproceedings{wu2026_4dnvs,
  title     = {4D Neural Voxel Splatting},
  author    = {Wu, Chun Tin and Chen, Jun Cheng},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}</div>
      </div>
    </div>
  </section>

  <!-- remaining sections go here -->
```

- [ ] **Step 2: Verify copy button works**

Reload `index.html`. Click "Copy". Expected: button text changes to "Copied!" for 2 seconds, then reverts. Paste into a text editor to confirm the BibTeX block was copied correctly.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add bibtex section with copy-to-clipboard"
```

---

### Task 7: Footer, final cleanup, and GitHub Pages activation

**Files:**
- Modify: `index.html` — replace remaining placeholder with footer; remove placeholder comment

- [ ] **Step 1: Insert footer**

Replace `<!-- remaining sections go here -->` with:

```html
  <!-- ── 7. FOOTER ── -->
  <footer>
    <p>
      This work builds on
      <a href="https://github.com/graphdeco-inria/gaussian-splatting" target="_blank" rel="noopener">3D Gaussian Splatting</a>,
      <a href="https://github.com/city-super/Scaffold-GS" target="_blank" rel="noopener">Scaffold-GS</a>,
      <a href="https://github.com/hustvl/4DGaussians" target="_blank" rel="noopener">4D Gaussians</a>, and
      <a href="https://github.com/Caoang327/HexPlane" target="_blank" rel="noopener">HexPlane</a>.
    </p>
    <p style="margin-top:6px;">
      Website design inspired by
      <a href="https://nerfies.github.io" target="_blank" rel="noopener">Nerfies</a>.
    </p>
  </footer>
```

- [ ] **Step 2: Full visual check**

Reload `index.html` in browser. Scroll top to bottom and verify all 7 sections render correctly:
1. Header — title, authors, affiliations, three buttons
2. Teaser video — plays silently, full-width gray band
3. Abstract — justified serif text
4. Method Overview — overview.png with caption
5. Supplementary Results — 2-column grid + full-width row
6. BibTeX — copy button works
7. Footer — acknowledgment links

Resize the window below 600px and confirm videos stack to single column.

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add footer and complete project page"
```

- [ ] **Step 4: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 5: Enable GitHub Pages**

Go to the repo on GitHub:  
`Settings → Pages → Source → Deploy from a branch → Branch: main / (root) → Save`

Expected: GitHub will display a URL like `https://tintin-wu.github.io/4d-neural-voxel-splatting/` within 1–2 minutes.

- [ ] **Step 6: Verify live page**

Open `https://tintin-wu.github.io/4d-neural-voxel-splatting/` in a browser. Confirm all sections render correctly and both active links (arXiv, Code) resolve.

> **Note:** Once the ICIP paper is published, update the `<span class="btn disabled">📄 Paper (Coming Soon)</span>` in `index.html` to `<a class="btn" href="<ICIP_URL>">📄 Paper</a>`.
