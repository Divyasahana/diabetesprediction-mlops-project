# Contributing Guidelines

Thank you for contributing to this project.

## Branching Strategy

- Do NOT push directly to main.
- Create a feature branch:
  git checkout -b feature/<task-name>
- Push branch:
  git push -u origin feature/<task-name>

## Pull Requests

- Open a Pull Request to main.
- Add description of changes.
- Wait for review before merging.

## Code Structure

- src/ → production code
- notebooks/ → experimentation only
- tests/ → unit tests
- Do not hardcode file paths.

## Dependencies

If adding new package:
- Update requirements.txt
