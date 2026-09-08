# Demo recording script (about 90 seconds)

## Preparation

1. Start the service with `run_lab.ps1` and wait for the health endpoint.
2. Open `http://127.0.0.1:8000/#inspector`.
3. Keep the Evaluations page available in a second tab.

## Recording

1. **Problem (10 seconds)** - Explain that a plausible answer is insufficient if the correct policy evidence was missed or reranked away.
2. **Normal cited answer (20 seconds)** - Ask the `$50 supplier gift card` question and point to the server-validated page 14 evidence.
3. **Retrieval trace (20 seconds)** - Run comparison with expected page `14`; show Dense, Hybrid candidates, Reranked evidence, and the diagnosis.
4. **Explicit abstention (15 seconds)** - Ask for the cafeteria Wi-Fi password; show the fixed insufficient-evidence response with no citation.
5. **Cross-page evidence (15 seconds)** - Ask the gift-card versus healthcare-provider comparison; show pages 14 and 16.
6. **Evaluation boundary (10 seconds)** - State that the frozen test has 8 hand-labelled questions, measures retrieval rather than answer accuracy, and that reranking improves order while increasing latency.

Do not claim production deployment, real users, answer accuracy, or AWS hosting.
