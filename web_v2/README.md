# Policy Workspace UI

The new API serves `index.html` at `/` and browser assets at `/static/v2/`. This is a plain HTML/CSS/JavaScript application with no frontend build step. The original `web/` application remains separate.

The interface defaults to Chinese and offers English. Reader, manager, and administrator views use the authenticated session returned by `/api/v2/session`. All authorization remains a server responsibility. Authentication uses the server's HttpOnly cookie; authenticated writes send `X-CSRF-Token`. No authentication token is stored in browser storage. Session storage holds only the language preference and the last update thread ID, keyed by account. Signing out removes rendered policy data from the page.

Document lists fetch 50 policies per page. Batch uploads send files sequentially with multipart XHR progress and display separate background preparation status. The upload limit is taken from the server. Failed uploads retry the whole file; browser upload resumption and direct-to-object-store multipart uploads are not implemented. Job polling runs every five seconds when the current list contains unfinished work, including an explicitly tracked publication job. A queued publication is never presented as a published revision.

Policy updates display clarification candidates with canonical previews, followed by a before/after review and explicit approval or rejection. The diff highlights the differing span; unchanged text between multiple edits can be included in that span. It is a review aid, not a semantic equivalence check. The authoritative proposal, concurrency checks, and publication happen on the server. Original attachments and exported canonical Markdown are separate links.

All document, model, and audit text is rendered through DOM text nodes. Rich HTML and model-generated Markdown are deliberately displayed as text. The trace panel exposes the backend trace on demand. Account administration supports role/scope changes and activation; password reset is outside this interface.

## Browser checks

`ui.test.cjs` uses Node's test runner and Playwright with a local static server and explicit API fixtures. It tests XSS handling, cookie/CSRF request behavior, role-specific views, clarification and approval, publication status, scoped uploads, pagination, administrative actions, failed retry, account-data clearing, and narrow-screen overflow. These are frontend contract tests, not backend/model-quality or capacity benchmarks.

Run `node --test web_v2/ui.test.cjs` with Playwright available on `NODE_PATH`. Set `POLICY_TEST_BROWSER` to a Chromium executable when Chrome is not at its usual Windows path.

`preview_check.cjs` is an optional smoke check against the repository's synthetic local preview backend. Set `POLICY_PREVIEW_URL`, `POLICY_PREVIEW_USERNAME`, and `POLICY_PREVIEW_PASSWORD`. An optional `POLICY_PREVIEW_SCREENSHOTS` directory receives screenshots. It verifies login, a seeded 500-unit lodging-policy answer, revision viewing, a real LangGraph proposal changing 500 to 800, reload recovery, and the 390px layout. It creates a review draft and stops before approval. Its controlled fixture model does not validate real-model accuracy.
