# PyLithics 2.0 JOSS Paper Draft

This directory contains the draft JOSS (Journal of Open Source Software) submission for PyLithics version 2.0.

## Files Included

- `paper.md` - Main paper in JOSS markdown format (977 words)
- `paper.bib` - Bibliography with all cited references in BibTeX format
- `README.md` - This file

## Paper Structure

The paper follows JOSS requirements and includes:

1. **YAML Metadata Header**
   - Title, tags, authors with ORCID IDs
   - Institutional affiliations
   - Bibliography reference

2. **Summary** (~150 words)
   - Overview of PyLithics 2.0 capabilities
   - Key innovations vs version 1.0
   - Research applications

3. **Statement of Need** (~250 words)
   - Archaeological context and research gaps
   - Methodological challenges addressed
   - How PyLithics 2.0 fills the need

4. **Methods and Architecture** (~400 words)
   - 11-step processing pipeline
   - Modular architecture description
   - Technical dependencies

5. **Results and Output** (~150 words)
   - CSV data structure
   - Visualization outputs
   - Voronoi diagrams

6. **Acknowledgements**
   - Funding and institutional support

## Key Features Highlighted (New in V2)

- Voronoi spatial distribution analysis
- Bilateral symmetry quantification
- Scar complexity/adjacency metrics
- Lateral edge convexity analysis
- Automated cortex detection
- DPI-aware arrow detection
- YAML-based configuration system
- Modular, customizable architecture
- CSV output format
- Per-lithic structured JSON export (`--export_json`)
- Interactive Streamlit dashboard (`--explore`) with thematic distribution tabs and per-lithic drill-downs

## Word Count

- **Total**: 977 words (within 1000 word JOSS limit ✓)

## Citations

All citations reference real academic publications:
- Archaeological methodology papers (Shea, Clarkson, Dibble, etc.)
- Lithic analysis references (Martingell, Inizan, etc.)
- Python dependencies (NumPy, SciPy, Pandas, OpenCV, Matplotlib, Shapely)
- PyLithics v1.0 paper (Gellis et al. 2021)

## What's Still Needed

Before JOSS submission, you'll need to:

1. **Figures**
   - [ ] Figure 1: Pipeline flowchart showing 11-step workflow
   - [ ] Figure 2: Example output showing annotated illustration with color-coded features
   - Create figures and save to this directory
   - Add figure references to `paper.md` using `\autoref{fig:label}` syntax

2. **Validation**
   - [ ] Verify all author information and affiliations are current
   - [ ] Check ORCID IDs are correct
   - [ ] Update date to submission date
   - [ ] Review all technical claims against current codebase

3. **Repository Preparation**
   - [ ] Ensure repository is public
   - [ ] Create release tag for version 2.0
   - [ ] Update main README with citation information
   - [ ] Verify LICENSE file is present (GNU GPLv3)
   - [ ] Add CONTRIBUTING.md if not present
   - [ ] Enable GitHub issues

4. **Testing PDF Compilation**
   - Test paper compiles correctly using JOSS Docker image:
     ```bash
     docker run --rm -v $PWD:/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref paper.md
     ```

5. **Final Review**
   - [ ] Proofread for typos and clarity
   - [ ] Verify all citations render correctly
   - [ ] Check figure references work
   - [ ] Ensure all claims are accurate and defensible

## Submission Process

1. Fork the JOSS reviews repository
2. Create submission issue with paper metadata
3. Wait for editor assignment
4. Respond to reviewer feedback
5. Make revisions as requested
6. Final acceptance and publication

## Notes

- Paper emphasizes major improvements over v1.0
- Focus on archaeological research applications and methodological rigor
- Modular architecture highlighted as key design feature
- All measurements and analyses are scientifically grounded
- References include both archaeological domain literature and technical dependencies
