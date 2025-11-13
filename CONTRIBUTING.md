# Contributing to Yuna Nami

Thank you for your interest in contributing to Yuna Nami! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Spam or excessive self-promotion
- Any conduct that could reasonably be considered inappropriate

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of asyncio and Telegram bots
- Familiarity with PyTorch (for ML contributions)

### Areas for Contribution

We welcome contributions in the following areas:

1. **🐛 Bug Fixes** - Fix issues and improve stability
2. **✨ New Features** - Add new capabilities
3. **🌐 Localization** - Add language support
4. **📚 Documentation** - Improve docs and examples
5. **🧪 Testing** - Write unit and integration tests
6. **🎨 UI/UX** - Improve user interaction patterns
7. **⚡ Performance** - Optimize algorithms and reduce latency
8. **🔒 Security** - Identify and fix security issues

---

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/YunaNami.git
cd YunaNami
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your test bot token
nano .env
```

### 5. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Check code formatting
black --check yuma.py

# Run linter
flake8 yuma.py --max-line-length=120
```

---

## 🛠️ How to Contribute

### 1. Choose an Issue

- Browse [open issues](https://github.com/0penAGI/YunaNami/issues)
- Look for issues tagged with `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it

### 2. Create a Branch

```bash
# Create a descriptive branch name
git checkout -b feature/add-spanish-support
git checkout -b bugfix/memory-leak-in-voice
git checkout -b docs/improve-installation-guide
```

### 3. Make Your Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add comments for complex logic
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_markov.py -v

# Check test coverage
pytest --cov=yuma tests/
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Write a descriptive commit message
git commit -m "feat: add Spanish language support to MultiLangLearner"
```

#### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(voice): add pitch modulation based on emotion
fix(memory): resolve SQLite connection leak
docs(readme): update installation instructions for Windows
```

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/add-spanish-support

# Go to GitHub and create a Pull Request
```

---

## 📏 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Maximum line length: 120 characters
# Use 4 spaces for indentation
# Use double quotes for strings (unless single quotes avoid escaping)

# Good
def calculate_resonance(message: dict) -> float:
    """Calculate resonance score for a message.
    
    Args:
        message: Dictionary containing message data
        
    Returns:
        Resonance score between 0.0 and 1.0
    """
    energy = message.get("energy", 0.0)
    return min(1.0, energy / 100.0)

# Bad
def calc_reso(msg):
    e=msg.get('energy',0.0)
    return min(1.0,e/100.0)
```

### Async Best Practices

```python
# Always use async/await for I/O operations
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Use asyncio.gather for concurrent operations
results = await asyncio.gather(
    fetch_data(url1),
    fetch_data(url2),
    fetch_data(url3)
)

# Add timeouts to prevent hanging
try:
    data = await asyncio.wait_for(fetch_data(url), timeout=10.0)
except asyncio.TimeoutError:
    logger.warning("Request timed out")
```

### Error Handling

```python
# Always handle exceptions appropriately
try:
    result = await risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Provide fallback or re-raise
except Exception as e:
    logger.exception("Unexpected error occurred")
    # Log full traceback for debugging
```

### Type Hints

```python
# Use type hints for function signatures
from typing import List, Dict, Optional, Union

async def process_words(
    text: str,
    lang: Optional[str] = None
) -> List[str]:
    """Process text and return word list."""
    words = text.split()
    return [w.lower() for w in words]

def update_chain(
    chain: Dict[str, List[str]],
    word: str,
    next_word: str
) -> None:
    """Update Markov chain with new transition."""
    chain.setdefault(word, []).append(next_word)
```

### Documentation

```python
# Use Google-style docstrings
async def train_resonance_model(
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 10
) -> float:
    """Train the neural resonance model.
    
    This function performs mini-batch gradient descent to optimize
    the resonance prediction model. It uses MSE loss and Adam optimizer.
    
    Args:
        features: Input feature tensor of shape (N, 10)
        targets: Target resonance scores of shape (N, 1)
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Final training loss as a float
        
    Raises:
        ValueError: If feature dimensions don't match model input
        RuntimeError: If CUDA is required but unavailable
        
    Example:
        >>> features = torch.randn(100, 10)
        >>> targets = torch.rand(100, 1)
        >>> loss = await train_resonance_model(features, targets)
        >>> print(f"Training loss: {loss:.4f}")
    """
    # Implementation here
    pass
```

---

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_markov.py
import pytest
from yuma import markov_chain, update_markov_chain

@pytest.fixture
def clean_chain():
    """Provide a fresh Markov chain for each test."""
    chain = {}
    yield chain
    chain.clear()

def test_markov_chain_update(clean_chain):
    """Test that Markov chain updates correctly."""
    update_markov_chain(clean_chain, "hello", "world")
    assert "hello" in clean_chain
    assert "world" in clean_chain["hello"]

@pytest.mark.asyncio
async def test_async_learning():
    """Test asynchronous word learning."""
    result = await MultiLangLearner.learn_word("test")
    assert result is not None
    assert "ja" in result
```

### Test Coverage Requirements

- New features must include tests
- Aim for >80% code coverage
- Test both success and failure cases
- Include edge cases and boundary conditions

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=yuma --cov-report=html

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"

# Run specific test class
pytest tests/test_agents.py::TestMultiAgentEngine
```

---

## 📝 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Screenshots (if applicable)
Add screenshots for UI/UX changes

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linters
2. **Code Review**: Maintainer reviews code quality and design
3. **Feedback**: Address review comments and push updates
4. **Approval**: Once approved, maintainer will merge

### Merge Criteria

- All CI checks pass
- At least one maintainer approval
- No unresolved conversations
- Branch is up-to-date with main

---

## 🐛 Issue Guidelines

### Creating Issues

Use the appropriate issue template:

**Bug Report:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Send message '...'
2. Bot responds with '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
 - OS: [e.g. Ubuntu 22.04]
 - Python version: [e.g. 3.10.6]
 - Bot version: [e.g. 3.2]

**Additional context**
Any other context about the problem
```

**Feature Request:**
```markdown
**Is your feature request related to a problem?**
A clear description of the problem

**Describe the solution you'd like**
A clear description of what you want to happen

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other context or screenshots
```

---

## 👥 Community

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Discord** (coming soon): Real-time chat with contributors
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)

### Recognition

Contributors will be:
- Listed in the README.md contributors section
- Mentioned in release notes for significant contributions
- Invited to join the core team (for regular contributors)

### Getting Help

If you need help:
1. Check the [README](README.md) and documentation
2. Search existing issues and discussions
3. Ask in GitHub Discussions
4. Tag maintainers for urgent issues

---

## 📜 License

By contributing to Yuna Nami, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Your contributions make Yuma Nami better for everyone. We appreciate your time and effort!

**Happy Coding! 🚀**
