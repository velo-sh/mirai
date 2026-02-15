# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Active  |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public issue**.

Instead, report it privately by emailing the maintainer directly. Include:

1. A description of the vulnerability.
2. Steps to reproduce.
3. The potential impact.
4. Any suggested fixes (optional).

We aim to acknowledge reports within **48 hours** and provide a fix or mitigation
within **7 days** for critical issues.

## Security Practices

- **Dependency scanning**: Automated via Dependabot for Python packages.
- **Static analysis**: Bandit runs in CI to detect common security issues.
- **Secret management**: API keys and tokens are loaded from environment variables
  or encrypted credential files — never committed to source control.
- **OAuth token handling**: Tokens are stored locally with restricted file
  permissions and refreshed automatically before expiry.

## Scope

The following are considered in scope:

- Authentication and authorization flaws
- Injection vulnerabilities (command injection via shell tool, etc.)
- Information disclosure through logs or error messages
- Credential / token exposure

The following are out of scope:

- Denial-of-service attacks on local development instances
- Vulnerabilities in third-party dependencies (report upstream)
