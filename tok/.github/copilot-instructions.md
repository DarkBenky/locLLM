# Go style conventions

- No comments unless critical for understanding non-obvious logic.
- For echo v5 handlers, bind request data using `c.Bind` with an inline struct and struct tags:
  - POST/JSON body: use `json:"field_name"` tags.
  - GET/query params: use `query:"field_name"` tags.
- Define the request struct inline inside the handler function, not at package level.
- Echo v5 handlers take `*echo.Context` (pointer), not `echo.Context`.
- Handle errors with early returns; avoid else chains.
