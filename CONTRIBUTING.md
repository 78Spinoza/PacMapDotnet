# Contributing to PacMapDotnet

We welcome contributions to PacMapDotnet! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

- **Bug Reports**: Use the [GitHub Issues](https://github.com/78Spinoza/PacMapDotnet/issues) page
- **Feature Requests**: Open an issue with the "enhancement" label
- **Questions**: Use GitHub Discussions for general questions

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/PacMapDotnet.git
   cd PacMapDotnet
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/78Spinoza/PacMapDotnet.git
   ```

3. **Install dependencies**
   ```bash
   # Initialize submodules (hnswlib)
   git submodule update --init --recursive

   # Build the solution
   dotnet build PACMAPCSharp.sln
   ```

4. **Run tests**
   ```bash
   dotnet test
   ```

## 📋 Contribution Types

### Code Contributions

#### C# Wrapper (PACMAPCSharp/)
- **New Features**: API improvements, additional distance metrics, enhanced error handling
- **Bug Fixes**: Memory leaks, incorrect parameter handling, platform-specific issues
- **Performance**: Optimizations, better memory management, parallel processing

#### C++ Implementation (pacmap_pure_cpp/)
- **Algorithm Improvements**: Better triplet sampling, optimization enhancements
- **Platform Support**: New architectures, OS-specific optimizations
- **Performance**: SIMD optimizations, memory improvements, GPU acceleration

#### Documentation
- **API Documentation**: Method documentation, parameter explanations
- **Examples**: Sample applications, use cases, tutorials
- **Performance Guides**: Benchmarking, optimization tips

## 🛠️ Development Guidelines

### Code Style

#### C# Code
- Follow [Microsoft C# coding conventions](https://docs.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions)
- Use regions to organize code logically
- Include XML documentation for all public APIs
- Use meaningful variable and method names

#### C++ Code
- Follow modern C++ standards (C++17+)
- Use RAII principles for resource management
- Include comprehensive error handling
- Document complex algorithms with comments

### Testing

#### Unit Tests
- All new features must include unit tests
- Test coverage should be >90%
- Use meaningful test names that describe what is being tested

#### Integration Tests
- Test C++/C# interop functionality
- Validate model persistence (save/load)
- Test cross-platform compatibility

#### Performance Tests
- Include benchmarks for significant algorithm changes
- Validate performance doesn't regress
- Test with various dataset sizes

## 📝 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Include tests for new functionality
   - Ensure all existing tests pass

3. **Test your changes**
   ```bash
   # Run all tests
   dotnet test

   # Run with coverage
   dotnet test --collect:"XPlat Code Coverage"

   # Run benchmarks
   cd benchmarks
   dotnet run --configuration Release
   ```

4. **Update documentation**
   - Update README.md if needed
   - Add/change API documentation
   - Update CHANGELOG.md

5. **Submit pull request**
   - Use descriptive title and description
   - Link to related issues
   - Include screenshots for UI changes
   - Explain the reasoning behind major changes

## 🧪 Code Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Performance**: Is it efficient? Does it scale well?
- **Maintainability**: Is the code readable and maintainable?
- **Documentation**: Is the code well-documented?
- **Testing**: Are there adequate tests?
- **Consistency**: Does it follow project conventions?

### Review Guidelines

- Be constructive and respectful in feedback
- Focus on the code, not the person
- Explain the reasoning behind suggestions
- Help improve the overall code quality

## 🐛 Bug Reporting

When reporting bugs, please include:

1. **Environment**: OS, .NET version, architecture
2. **Reproduction Steps**: Clear steps to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Sample Code**: Minimal code that reproduces the issue
6. **Stack Trace**: Any error messages or stack traces

## 💡 Feature Requests

When requesting features:

1. **Use Case**: Explain the problem you're trying to solve
2. **Proposed Solution**: Describe the desired functionality
3. **Alternatives**: Mention any alternative solutions considered
4. **Impact**: How would this feature benefit users?

## 🔧 Development Tools

### Required Tools
- **.NET 8.0 SDK**: For C# development
- **Visual Studio 2022** or **VS Code**: For development
- **Git**: For version control
- **CMake**: For C++ builds (if modifying native code)

### Recommended Tools
- **GitKraken** or **SourceTree**: Git GUI clients
- **JetBrains Rider**: Advanced C# IDE
- **CLion**: For C++ development
- **Postman**: For testing API endpoints

## 📊 Performance Guidelines

### Benchmarking
- Use the built-in benchmark suite
- Test with various dataset sizes
- Compare against baseline performance
- Document performance changes

### Memory Management
- Monitor memory usage with large datasets
- Use memory profiling tools
- Ensure proper cleanup of resources
- Validate no memory leaks

## 🌐 Platform Support

### Target Platforms
- **Windows x64**: Primary development platform
- **Linux x64**: CI/CD and production deployments

### Testing Across Platforms
- Windows: Visual Studio tests
- Linux: Docker-based CI/CD pipeline
- Ensure consistent behavior across platforms

## 📋 Release Process

### Version Numbers
- Follow [Semantic Versioning](https://semver.org/)
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass on all platforms
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Performance benchmarks are run
- [ ] Security review (if applicable)
- [ ] Version number is updated
- [ ] Release notes are prepared

## 🤖 Automation

### CI/CD Pipeline
- **Build**: Automated builds on PRs
- **Test**: Run all tests on multiple platforms
- **Coverage**: Generate and track code coverage
- **Performance**: Run benchmark suite
- **Security**: Security scanning for dependencies

### Quality Gates
- Minimum 90% test coverage
- All tests must pass
- No critical security vulnerabilities
- Performance regression checks

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions
- **Email**: support@pacmapdotnet.com
- **Discord**: [Community server link]

## 🙏 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Invited to the contributor team
- Eligible for contributor swag

Thank you for contributing to PacMapDotnet! 🎉