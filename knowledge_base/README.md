# Knowledge Base Editing Guide
# git commit: docs(kb): add knowledge base editing guide
# Module: knowledge-base

# Life Insurance Knowledge Base — Editing Guide

Welcome! This directory contains the knowledge base for the Life Insurance Support Assistant.
**You do NOT need to be a programmer** to edit these files.

## How It Works

Each `.yaml` file contains information about a specific life insurance topic. The AI agent
reads these files to answer user questions accurately.

## File Format

Every YAML file follows this structure:

```yaml
category: "policy_types"          # Which category this belongs to
topic: "Term Life Insurance"      # The main topic title
keywords:                         # Words that help find this content
  - "term life"
  - "temporary coverage"

summary: |                        # A brief overview (1-3 sentences)
  Term life insurance provides coverage for a specific period.

details:                          # Detailed information (flexible structure)
  definition: |
    A longer explanation...
  key_features:
    - "Feature 1"
    - "Feature 2"

related_topics:                   # Links to other topics
  - "whole_life"
  - "universal_life"
```

## Categories

| Category | Directory | What It Covers |
|----------|-----------|---------------|
| `policy_types` | `policies/` | Types of life insurance policies |
| `claims` | `claims/` | Filing claims, documents, status |
| `eligibility` | `eligibility/` | Age, health, underwriting |
| `benefits` | `benefits/` | Death benefit, cash value, riders, tax |
| `faq` | `faq/` | General questions, payments, changes |

## How to Edit

1. Open any `.yaml` file in a text editor (Notepad, VS Code, etc.)
2. Modify the content you want to update
3. Save the file
4. Reload the knowledge base:
   - API: `POST /api/v1/knowledge-base/reload`
   - CLI: Run `make index-kb`

## Rules

- **Keep the `category` and `topic` fields** — don't remove them
- **Use proper YAML syntax** — indentation matters (use 2 spaces, not tabs)
- **Multi-line text** — use `|` after the key for paragraphs
- **Lists** — start each item with `- ` (dash + space)
- **Test after editing** — reload and ask a question about your change
