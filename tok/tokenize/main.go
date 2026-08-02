package main

import (
	"crypto/sha256"
	"encoding/hex"
	"log"
	"net/http"
	"strings"
	"sync"
	"tokenize/libs"

	"github.com/labstack/echo/v5"
	"github.com/labstack/echo/v5/middleware"
)

type CategoryStore struct {
	mu         sync.Mutex
	Categories map[string]uint8
}

var tok *libs.Tokenizer
var db *libs.DB
var categoryStore = CategoryStore{
	Categories: make(map[string]uint8),
}

func init() {
	tok = libs.NewTokenizer("tokenizer_models/tokenizer.model")
	log.Printf("Vocab size: %d", tok.GetVocabSize())

	testString := "testing if tokenizer works"
	rec := tok.Tokenize(testString, 0)
	out := tok.GetText(rec)
	out = strings.TrimSuffix(out, "</s>")
	if out != testString {
		log.Fatalf("tokenizer does not work: input=%q output=%q", testString, out)
	}

	var err error
	db, err = libs.Open("db/tokenize.db", "db/tokenize.bin")
	if err != nil {
		log.Fatal(err)
	}

	categoryMap, err := db.GetCategoryIndex()
	if err != nil {
		log.Fatal(err)
	}
	categoryStore.Categories = categoryMap
}

func handleReceiveData(c *echo.Context) error {
	type Request struct {
		Text     string `json:"text"`
		Category string `json:"category"`
	}

	req := new(Request)
	if err := c.Bind(req); err != nil {
		return err
	}

	// Deduplication: skip if text hash already exists
	hashBytes := sha256.Sum256([]byte(req.Text))
	contentHash := hex.EncodeToString(hashBytes[:])

	exists, err := db.HashExists(contentHash)
	if err != nil {
		return err
	}
	if exists {
		log.Printf("Skipping duplicate: %s...", req.Text[:min(len(req.Text), 60)])
		return c.JSON(http.StatusOK, map[string]interface{}{
			"tokenizer_vocab_size": tok.GetVocabSize(),
			"token_count":          0,
			"duplicate":            true,
		})
	}

	log.Printf("Received text: %s", req.Text)
	log.Printf("Received category: %s", req.Category)

	categoryStore.mu.Lock()
	id, ok := categoryStore.Categories[req.Category]
	categoryStore.mu.Unlock()

	if !ok {
		idx, err := db.GetOrCreateCategory(req.Category)
		if err != nil {
			return err
		}
		categoryStore.mu.Lock()
		categoryStore.Categories[req.Category] = idx
		categoryStore.mu.Unlock()
		id = idx
	}

	rec := tok.Tokenize(req.Text, id)

	if err := db.SaveRecord(rec, contentHash); err != nil {
		return err
	}
	return c.JSON(http.StatusOK, map[string]int{"tokenizer_vocab_size": tok.GetVocabSize(), "token_count": int(rec.TokenCount)})
}

func handleGetNextSamples(c *echo.Context) error {
	type Request struct {
		SampleCount int `query:"sample_count"`
	}

	req := new(Request)
	if err := c.Bind(req); err != nil {
		return err
	}
	if req.SampleCount <= 0 {
		req.SampleCount = 256
	}

	samples, err := db.GetNextSamples(req.SampleCount)
	if err != nil {
		return err
	}
	if samples == nil {
		samples = [][]byte{}
	}
	return c.JSON(http.StatusOK, map[string]interface{}{"samples": samples})
}

func handleGetNextSamplesRandom(c *echo.Context) error {
	type Request struct {
		SampleCount int `query:"sample_count"`
	}

	req := new(Request)
	if err := c.Bind(req); err != nil {
		return err
	}
	if req.SampleCount <= 0 {
		req.SampleCount = 256
	}

	samples, err := db.GetRandomSamples(req.SampleCount)
	if err != nil {
		return err
	}
	if samples == nil {
		samples = [][]byte{}
	}
	return c.JSON(http.StatusOK, map[string]interface{}{"samples": samples})
}

func handleGetCategoryIndex(c *echo.Context) error {
	categoryStore.mu.Lock()
	defer categoryStore.mu.Unlock()
	return c.JSON(http.StatusOK, categoryStore.Categories)
}

func handleResetCursor(c *echo.Context) error {
	if err := db.ResetCursor(); err != nil {
		return err
	}
	return c.JSON(http.StatusOK, map[string]string{"status": "cursor reset to 0"})
}

func handleStats(c *echo.Context) error {
	stats, err := db.GetStats()
	if err != nil {
		return err
	}
	return c.JSON(http.StatusOK, stats)
}

func handleStatsGUI(c *echo.Context) error {
	stats, err := db.GetStats()
	if err != nil {
		return err
	}
	return c.HTML(http.StatusOK, libs.RenderStatsHTML(stats))
}

func main() {
	e := echo.New()

	e.Use(middleware.RequestLogger())
	e.Use(middleware.Recover())

	e.POST("/api/receive-data", handleReceiveData)
	e.GET("/api/get-next-samples", handleGetNextSamples)
	e.GET("/api/get-next-samples-random", handleGetNextSamplesRandom)
	e.GET("/api/get-category-index", handleGetCategoryIndex)
	e.POST("/api/reset-cursor", handleResetCursor)
	e.GET("/api/stats", handleStats)
	e.GET("/api/stats/gui", handleStatsGUI)
	if err := e.Start(":8823"); err != nil {
		log.Fatal(err)
	}
}
