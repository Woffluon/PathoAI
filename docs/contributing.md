# Contributing to PathoAI

We welcome contributions to PathoAI! This document provides guidelines for contributing code, documentation, and bug reports.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/woffluon/PathoAI.git
   cd PathoAI
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements/development.txt
   ```

## Development Guidelines

### Code Style

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting (line length: 100)
- Use **isort** for import sorting
- Add **type hints** for all function signatures
- Write **docstrings** for all public functions and classes

**Format code before committing**:
```bash
black --line-length 100 .
isort --profile black .
```

### Testing

- Write **unit tests** for new functionality
- Ensure **existing tests pass** before submitting PR
- Aim for **80%+ code coverage**

**Run tests**:
```bash
pytest tests/
```

### Documentation

- Update **docstrings** for code changes
- Update **README.md** if adding features
- Add **examples** for new APIs
- Update **CHANGELOG.md** with your changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add batch processing support
fix: Resolve memory leak in session cleanup
docs: Update installation instructions
refactor: Simplify image preprocessing pipeline
test: Add unit tests for watershed segmentation
```

**Commit message format**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

## Pull Request Process

1. **Update documentation** for your changes
2. **Add tests** for new functionality
3. **Run linters and tests** locally
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request** on GitHub
6. **Respond to review feedback** promptly

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] PR description explains changes

## Reporting Bugs

**Before reporting**:
- Check if the bug is already reported in [Issues](https://github.com/Woffluon/PathoAI/issues)
- Verify the bug exists in the latest version

**Bug report should include**:
- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, dependency versions
- **Logs**: Relevant error messages or stack traces

## Feature Requests

We welcome feature requests! Please:
- Check if the feature is already requested
- Describe the use case and benefits
- Provide examples of how it would work
- Consider implementation complexity

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.

---

*Thank you for contributing to PathoAI! - Efe Arabacı (@woffluon)*
