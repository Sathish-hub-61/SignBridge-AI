# Human-Rating Rubric Template

Template for human evaluation of SignBridge AI outputs. **Do not fill this in yourself** —
the project owner should complete it after reviewing real outputs.

Rate each item on a 1–5 scale (1 = poor, 5 = excellent) and add comments.

| # | Gloss sequence | Interpreted sentence | Intent | Clarity | Faithfulness to signs | Appropriateness | Comments |
|---|----------------|----------------------|--------|---------|-----------------------|-----------------|----------|
| 1 |                |                      |        |         |                       |                 |          |
| 2 |                |                      |        |         |                       |                 |          |
| 3 |                |                      |        |         |                       |                 |          |
| 4 |                |                      |        |         |                       |                 |          |
| 5 |                |                      |        |         |                       |                 |          |

## Rating dimensions

- **Clarity:** Is the English sentence easy to understand?
- **Faithfulness to signs:** Does the sentence accurately reflect the recognized glosses (no invented content)?
- **Appropriateness:** Is the reply tone and content suitable for the intent (especially emergency)?

## Notes
- Evaluate on real collected ISL data, not the synthetic smoke dataset.
- Include low-confidence and ambiguous cases to test human-in-the-loop confirm flow.
- Rate the confirm/edit flow: can a user correct a wrong token before sending?
