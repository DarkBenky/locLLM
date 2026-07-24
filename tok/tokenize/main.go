package main

import (
	"log"
	"tokenize/libs"
)

func main() {
	t := libs.NewTokenizer("tokenizer_models/tokenizer.model")
	log.Printf("Vocab size: %d", t.GetVocabSize())

	text := "Hello, world! This is a test of the Llama 2 tokenizer."
	rec := t.Tokenize(text, 0)

	log.Printf("Input:   %q", text)
	log.Printf("Tokens:  %v (%d tokens)", rec.Tokens, len(rec.Tokens))

	decoded := t.GetText(rec)
	log.Printf("Decoded: %q", decoded)
}
