# Contributing to RDT 2

We welcome contributions, improvements, and modifications of all kinds. Everyone is free to use **RDT 2** in accordance with the [license](LICENSE). You are also encouraged to submit bug reports, feature requests, or pull requests. While we cannot guarantee that every pull request will be approved—our team is small and has limited bandwidth—we will make our best effort to review contributions. The following sections describe the process in detail.

## Issues and Feature Requests

For general questions or discussions that are not strictly bug reports or feature requests, please use the GitHub [Discussions](https://github.com/thu-ml/RDT2/discussions) page. This is a good place for asking about usage, sharing ideas, or raising broader topics.

If you discover a bug or another issue, please first search existing GitHub Issues to confirm that it has not already been reported. If it is new, include the following information when creating your issue:

* Your operating system and version, along with the Python version you are using
* Minimal code that reproduces the bug, including all dependencies
* The complete traceback of any exceptions
* Any additional context that may help (e.g., screenshots)

Reproducibility is critical. If you encountered the issue after modifying **RDT 2**, please attempt to reproduce it on an unmodified copy of `main` and share a code snippet that allows us to quickly replicate the problem.

For feature requests, please first check whether a similar request already exists. If not, please provide:

* The motivation behind the feature
* The problem you are trying to solve or your use case
* A clear description of the proposed feature
* How you expect to use it in practice

While we cannot commit to implementing every feature request, understanding your use cases helps us prioritize future development.

## Submitting a Pull Request

We welcome pull requests (PRs) for new robots, environments, or other features. Before starting, we recommend opening a [feature request](https://github.com/thu-ml/RDT2/issues) or starting a [discussion](https://github.com/thu-ml/RDT2/discussions) to get feedback on whether your proposal is likely to be merged.

Because we are a small team with limited capacity for maintenance and support, not all PRs can be accepted—for example, if a contribution makes the codebase significantly harder to maintain or falls outside our scope. Getting in touch early is the best way to ensure your work aligns with our roadmap. Even if a PR is not merged, you are always welcome to maintain your own fork with custom modifications.

When preparing a PR, please ensure that you:

* Provide a clear and descriptive title and summary
* Run `pre-commit` (after installing via `pre-commit install`)
* Run `ruff check .` and `ruff format .`
* Verify that all tests pass

