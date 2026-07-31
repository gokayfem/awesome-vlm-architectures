# Contributing
First of all, your contributions are always welcome!

This list provides architecture-focused explanations of significant Vision-Language Models (VLMs) and their primary sources. A suggested model should:

- Introduce a distinct architectural, training, alignment, or data contribution that can be explained precisely.
- Have a primary technical source, such as a paper, technical report, official model card, or maintained repository.
- Fill a genuine gap in the catalog rather than adding another checkpoint or parameter scale from an existing family.
- Be documented well enough for readers to trace every substantive claim to an original source.

## Guidelines
To ensure your pull request is accepted, please follow these guidelines:

- Verify if your suggestion has already been submitted to prevent redundancy.
- Historical architectures are welcome when they remain important to the field. Current projects should have maintained documentation; archived or superseded releases must be identified as historical.
- Submit one pull request per suggestion for clarity.
- Contributions that introduce new categories or refine the existing framework are encouraged.
- Documentation should be in English.
- Proofread your submission to avoid spelling or grammatical errors.
- Adjust your text editor settings to remove trailing whitespace.
- While we appreciate contributions, we advise against submitting your own projects. It's preferable to have someone else recognize the value of your work and recommend it.

## Entry requirements

Keep the established three-layer format: a short architecture-first summary, primary-source badges and authors, followed by an expandable deep dive.

- Use one entry per architecture family. Fold point releases and new parameter sizes into the family unless the architecture changes materially.
- Record the first public date as `**Released:** YYYY-MM-DD`. Prefer the first official model release; when that date is unavailable, use the paper's arXiv v1 submission or the first official technical report.
- Order badges as Paper, Code, then Model or Demo. Labels must match their destinations.
- Describe the mechanism before benchmark results. Attribute time-sensitive performance claims to the authors and publication date.
- Organize the expanded explanation as architecture, training or alignment, then datasets.
- Link primary sources directly. Secondary articles are appropriate only in the Important References section.
- Do not add proprietary models whose architecture is not disclosed in enough detail for a technical entry.

## Pull requests
1. Fork it!
2. Create your branch: git checkout -b my-new-branch
3. Commit your changes: git commit -am 'fix stuff'
4. Push to the branch: git push origin my-new-branch
5. Submit a pull request
