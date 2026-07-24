# Tokenizer Service

## APIs

- **POST /api/receive-data** => {text : "sample text", category : "example_category"} => {tokenizer_vocab_size: 32 000, token_count: 24}

1. Tokenize given text
2. Save to file

``` go
type Record struct {
    RecordSize uint64
    Category   uint8
    Tokens     []uint16
}
```

3. Save to Sqlite to 3 tables
3.1 If Category not in Category Table save new category to sql lite
3.2 Save to Stats table where we increment category count and category token count
3.3 Save to Records table where we save Category Token count and Index in the file
4. Respond to client with tokenizer vocab size and number of tokens for given text

- **GET /api/stats** => {} => {total_samples: 1024, total_tokens: 512000, categories: [{category: "c_programming", sample_count: 300, token_count: 150000}, {category: "general", sample_count: 724, token_count: 362000}], category_served_count: {"go_programming": 10, "python_programming": 20, ...}, category_served_tok_count: {"go_programming": 5000, "python_programming": 9800, ...}, current_file_index: 8192}

- **GET /api/get-next-samples** => {sample_count: 256} => {samples : [
    blobA
    blobB,
    ...
]}

1. Read Cursor table for last served file offset (resume point)
2. Read `sample_count` records from binary file starting there
3. Update Served table: increment served_count and served_token_count per category
4. Update Cursor table with new offset so next call resumes automatically

- **GET /api/get-category-index** => {} => {"c_programming": 0, "general": 1, ...}