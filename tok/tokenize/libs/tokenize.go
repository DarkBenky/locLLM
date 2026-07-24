package libs

import "github.com/lwch/sentencepiece"

type Tokenizer struct {
	sp *sentencepiece.Model
}

func NewTokenizer(modelPath string) *Tokenizer {
	sp, err := sentencepiece.Load(modelPath)
	if err != nil {
		panic(err)
	}
	return &Tokenizer{sp: sp}
}

type Record struct {
	TokenCount uint32
	Category   uint8
	Tokens     []uint16
}

func (t *Tokenizer) Tokenize(text string, category uint8) *Record {
	ids := t.sp.Encode(text, false, true)
	tokens := make([]uint16, len(ids))
	for i, id := range ids {
		tokens[i] = uint16(id)
	}
	return &Record{
		TokenCount: uint32(len(tokens)),
		Category:   category,
		Tokens:     tokens,
	}
}

func (t *Tokenizer) GetVocabSize() int {
	return t.sp.Count()
}

func (t *Tokenizer) GetText(record *Record) string {
	ids := make([]uint64, len(record.Tokens))
	for i, tok := range record.Tokens {
		ids[i] = uint64(tok)
	}
	return t.sp.Decode(ids)
}
