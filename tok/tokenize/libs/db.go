package libs

import (
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

var errRecordPastEOF = errors.New("record extends past EOF")

type DB struct {
	muDB     sync.Mutex
	db       *sql.DB
	muFile   sync.Mutex
	file     *os.File
	filePath string
}

type CategoryStats struct {
	Category    string `json:"category"`
	SampleCount int    `json:"sample_count"`
	TokenCount  int    `json:"token_count"`
}

type StatsResult struct {
	TotalSamples           int             `json:"total_samples"`
	TotalTokens            int             `json:"total_tokens"`
	Categories             []CategoryStats `json:"categories"`
	CategoryServedCount    map[string]int  `json:"category_served_count"`
	CategoryServedTokCount map[string]int  `json:"category_served_tok_count"`
	CurrentFileIndex       int64           `json:"current_file_index"`
}

const maxRecordSize = 4 * 1024 * 1024

const schema = `
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS stats (
    category_id INTEGER PRIMARY KEY REFERENCES categories(id),
    sample_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    token_count INTEGER NOT NULL,
    file_offset INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS served (
    category_id INTEGER PRIMARY KEY REFERENCES categories(id),
    served_count INTEGER NOT NULL DEFAULT 0,
    served_token_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    file_offset INTEGER NOT NULL DEFAULT 0
);
`

func Open(dbPath, filePath string) (*DB, error) {
	// Ensure parent directories exist (survives manual db/ deletion)
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("create db dir: %w", err)
	}

	sqlDB, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if err := sqlDB.Ping(); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("ping sqlite: %w", err)
	}
	if _, err := sqlDB.Exec(schema); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	if _, err := sqlDB.Exec("INSERT OR IGNORE INTO cursor (id, file_offset) VALUES (1, 0)"); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("seed cursor: %w", err)
	}

	// Migration: add content_hash column for deduplication (ignore error if column exists)
	sqlDB.Exec("ALTER TABLE records ADD COLUMN content_hash TEXT")
	sqlDB.Exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_records_content_hash ON records(content_hash)")

	file, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("open binary file: %w", err)
	}

	return &DB{
		db:       sqlDB,
		file:     file,
		filePath: filePath,
	}, nil
}

func (d *DB) Close() error {
	var errs []error
	if d.file != nil {
		if err := d.file.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close file: %w", err))
		}
	}
	if d.db != nil {
		if err := d.db.Close(); err != nil {
			errs = append(errs, fmt.Errorf("close db: %w", err))
		}
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

func (d *DB) GetOrCreateCategory(name string) (uint8, error) {
	d.muDB.Lock()
	defer d.muDB.Unlock()

	var id uint8
	err := d.db.QueryRow("SELECT id FROM categories WHERE name = ?", name).Scan(&id)
	if err == nil {
		return id, nil
	}
	if err != sql.ErrNoRows {
		return 0, fmt.Errorf("lookup category: %w", err)
	}

	var count int
	if err := d.db.QueryRow("SELECT COUNT(*) FROM categories").Scan(&count); err != nil {
		return 0, fmt.Errorf("count categories: %w", err)
	}
	if count >= 256 {
		return 0, fmt.Errorf("category limit reached (256)")
	}

	result, err := d.db.Exec("INSERT INTO categories (name) VALUES (?)", name)
	if err != nil {
		return 0, fmt.Errorf("insert category: %w", err)
	}
	rowID, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("last insert id: %w", err)
	}
	return uint8(rowID), nil
}

func (d *DB) GetCategoryIndex() (map[string]uint8, error) {
	d.muDB.Lock()
	defer d.muDB.Unlock()

	rows, err := d.db.Query("SELECT id, name FROM categories")
	if err != nil {
		return nil, fmt.Errorf("query categories: %w", err)
	}
	defer rows.Close()

	index := make(map[string]uint8)
	for rows.Next() {
		var id uint8
		var name string
		if err := rows.Scan(&id, &name); err != nil {
			return nil, fmt.Errorf("scan category: %w", err)
		}
		index[name] = id
	}
	return index, rows.Err()
}

func (d *DB) HashExists(hash string) (bool, error) {
	d.muDB.Lock()
	defer d.muDB.Unlock()

	var count int
	err := d.db.QueryRow("SELECT COUNT(*) FROM records WHERE content_hash = ?", hash).Scan(&count)
	return count > 0, err
}

func (d *DB) ResetCursor() error {
	d.muDB.Lock()
	defer d.muDB.Unlock()

	_, err := d.db.Exec("UPDATE cursor SET file_offset = 0 WHERE id = 1")
	return err
}

func (d *DB) SaveRecord(record *Record, contentHash string) error {
	d.muFile.Lock()
	d.muDB.Lock()
	defer d.muDB.Unlock()
	defer d.muFile.Unlock()

	offset, err := d.writeRecord(record)
	if err != nil {
		return fmt.Errorf("write record to file: %w", err)
	}

	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		"INSERT OR IGNORE INTO stats (category_id, sample_count, token_count) VALUES (?, 0, 0)",
		record.Category,
	); err != nil {
		return fmt.Errorf("ensure stats row: %w", err)
	}

	if _, err := tx.Exec(
		"UPDATE stats SET sample_count = sample_count + 1, token_count = token_count + ? WHERE category_id = ?",
		record.TokenCount, record.Category,
	); err != nil {
		return fmt.Errorf("update stats: %w", err)
	}

	if _, err := tx.Exec(
		"INSERT INTO records (category_id, token_count, file_offset, content_hash) VALUES (?, ?, ?, ?)",
		record.Category, record.TokenCount, offset, contentHash,
	); err != nil {
		return fmt.Errorf("insert record: %w", err)
	}

	if _, err := tx.Exec(
		"INSERT OR IGNORE INTO served (category_id, served_count, served_token_count) VALUES (?, 0, 0)",
		record.Category,
	); err != nil {
		return fmt.Errorf("ensure served row: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}
	return nil
}

func (d *DB) GetStats() (*StatsResult, error) {
	d.muDB.Lock()
	defer d.muDB.Unlock()

	result := &StatsResult{
		CategoryServedCount:    make(map[string]int),
		CategoryServedTokCount: make(map[string]int),
	}

	rows, err := d.db.Query(`
		SELECT c.name, COALESCE(s.sample_count,0), COALESCE(s.token_count,0),
		       COALESCE(sv.served_count,0), COALESCE(sv.served_token_count,0)
		FROM categories c
		LEFT JOIN stats s ON s.category_id = c.id
		LEFT JOIN served sv ON sv.category_id = c.id
	`)
	if err != nil {
		return nil, fmt.Errorf("query stats: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var name string
		var sampleCount, tokenCount, servedCount, servedTokenCount int
		if err := rows.Scan(&name, &sampleCount, &tokenCount, &servedCount, &servedTokenCount); err != nil {
			return nil, fmt.Errorf("scan stats row: %w", err)
		}
		result.TotalSamples += sampleCount
		result.TotalTokens += tokenCount
		result.Categories = append(result.Categories, CategoryStats{
			Category:    name,
			SampleCount: sampleCount,
			TokenCount:  tokenCount,
		})
		if servedCount > 0 || servedTokenCount > 0 {
			result.CategoryServedCount[name] = servedCount
			result.CategoryServedTokCount[name] = servedTokenCount
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate stats: %w", err)
	}

	fi, err := d.file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat file: %w", err)
	}
	result.CurrentFileIndex = fi.Size()

	return result, nil
}

func (d *DB) GetNextSamples(count int) ([][]byte, error) {
	d.muFile.Lock()
	d.muDB.Lock()
	defer d.muDB.Unlock()
	defer d.muFile.Unlock()

	var cursorOffset int64
	if err := d.db.QueryRow("SELECT file_offset FROM cursor WHERE id = 1").Scan(&cursorOffset); err != nil {
		return nil, fmt.Errorf("read cursor: %w", err)
	}

	fi, err := d.file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat file: %w", err)
	}
	fileSize := fi.Size()

	type parsedRecord struct {
		raw        []byte
		category   uint8
		tokenCount int
	}

	var parsed []parsedRecord
	offset := cursorOffset

	for i := 0; i < count; i++ {
		if offset >= fileSize {
			if fileSize == 0 {
				break
			}
			offset = 0 // wrap around for next epoch
		}
		raw, cat, tokCount, nextOff, err := readRecordAt(d.file, offset)
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF || err == errRecordPastEOF {
				break
			}
			return nil, fmt.Errorf("read record at %d: %w", offset, err)
		}
		parsed = append(parsed, parsedRecord{raw: raw, category: cat, tokenCount: int(tokCount)})
		offset = nextOff
	}

	if len(parsed) == 0 {
		return nil, nil
	}

	tx, err := d.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	servedByCategory := make(map[uint8][2]int)
	for _, p := range parsed {
		cur := servedByCategory[p.category]
		cur[0]++
		cur[1] += p.tokenCount
		servedByCategory[p.category] = cur
	}

	for catID, counts := range servedByCategory {
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO served (category_id, served_count, served_token_count) VALUES (?, 0, 0)",
			catID,
		); err != nil {
			return nil, fmt.Errorf("ensure served row for %d: %w", catID, err)
		}
		if _, err := tx.Exec(
			"UPDATE served SET served_count = served_count + ?, served_token_count = served_token_count + ? WHERE category_id = ?",
			counts[0], counts[1], catID,
		); err != nil {
			return nil, fmt.Errorf("update served for %d: %w", catID, err)
		}
	}

	if _, err := tx.Exec("UPDATE cursor SET file_offset = ? WHERE id = 1", offset); err != nil {
		return nil, fmt.Errorf("update cursor: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit tx: %w", err)
	}

	samples := make([][]byte, len(parsed))
	for i, p := range parsed {
		samples[i] = p.raw
	}
	return samples, nil
}

func (d *DB) GetRandomSamples(count int) ([][]byte, error) {
	d.muDB.Lock()
	var maxID int64
	err := d.db.QueryRow("SELECT COALESCE(MAX(id), 0) FROM records").Scan(&maxID)
	d.muDB.Unlock()
	if err != nil {
		return nil, fmt.Errorf("read max id: %w", err)
	}
	if maxID == 0 {
		return nil, nil
	}

	d.muFile.Lock()
	d.muDB.Lock()
	defer d.muDB.Unlock()
	defer d.muFile.Unlock()

	type parsedRecord struct {
		raw        []byte
		category   uint8
		tokenCount int
	}

	var parsed []parsedRecord
	for i := 0; i < count; i++ {
		randID := rand.Int63n(maxID) + 1
		var offset int64
		err := d.db.QueryRow(
			"SELECT file_offset FROM records WHERE id >= ? LIMIT 1", randID,
		).Scan(&offset)
		if err != nil {
			if err == sql.ErrNoRows {
				continue
			}
			return nil, fmt.Errorf("query record offset: %w", err)
		}

		raw, cat, tokCount, _, err := readRecordAt(d.file, offset)
		if err != nil {
			if err == errRecordPastEOF || err == io.EOF || err == io.ErrUnexpectedEOF {
				continue
			}
			return nil, fmt.Errorf("read record at %d: %w", offset, err)
		}
		parsed = append(parsed, parsedRecord{raw: raw, category: cat, tokenCount: int(tokCount)})
	}

	if len(parsed) == 0 {
		return nil, nil
	}

	tx, err := d.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	servedByCategory := make(map[uint8][2]int)
	for _, p := range parsed {
		cur := servedByCategory[p.category]
		cur[0]++
		cur[1] += p.tokenCount
		servedByCategory[p.category] = cur
	}

	for catID, counts := range servedByCategory {
		if _, err := tx.Exec(
			"INSERT OR IGNORE INTO served (category_id, served_count, served_token_count) VALUES (?, 0, 0)",
			catID,
		); err != nil {
			return nil, fmt.Errorf("ensure served row for %d: %w", catID, err)
		}
		if _, err := tx.Exec(
			"UPDATE served SET served_count = served_count + ?, served_token_count = served_token_count + ? WHERE category_id = ?",
			counts[0], counts[1], catID,
		); err != nil {
			return nil, fmt.Errorf("update served for %d: %w", catID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit tx: %w", err)
	}

	samples := make([][]byte, len(parsed))
	for i, p := range parsed {
		samples[i] = p.raw
	}
	return samples, nil
}

func (d *DB) writeRecord(record *Record) (int64, error) {
	recordSize := uint64(1 + 2*len(record.Tokens))

	offset, err := d.file.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, fmt.Errorf("seek end: %w", err)
	}

	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], recordSize)
	if _, err := d.file.Write(buf[:]); err != nil {
		return 0, fmt.Errorf("write record size: %w", err)
	}

	if _, err := d.file.Write([]byte{record.Category}); err != nil {
		return 0, fmt.Errorf("write category: %w", err)
	}

	tokenBuf := make([]byte, 2*len(record.Tokens))
	for i, t := range record.Tokens {
		binary.LittleEndian.PutUint16(tokenBuf[2*i:], t)
	}
	if _, err := d.file.Write(tokenBuf); err != nil {
		return 0, fmt.Errorf("write tokens: %w", err)
	}

	if err := d.file.Sync(); err != nil {
		return 0, fmt.Errorf("sync file: %w", err)
	}

	return offset, nil
}

func readRecordAt(file *os.File, offset int64) (raw []byte, category uint8, tokenCount uint32, nextOffset int64, err error) {
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("seek: %w", err)
	}

	var sizeBuf [8]byte
	if _, err := io.ReadFull(file, sizeBuf[:]); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("read record size: %w", err)
	}
	recordSize := binary.LittleEndian.Uint64(sizeBuf[:])
	if recordSize < 1 || recordSize > maxRecordSize {
		return nil, 0, 0, 0, fmt.Errorf("invalid record size: %d", recordSize)
	}

	fi, err := file.Stat()
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("stat file: %w", err)
	}
	if offset+8+int64(recordSize) > fi.Size() {
		return nil, 0, 0, 0, errRecordPastEOF
	}

	payload := make([]byte, recordSize)
	if _, err := io.ReadFull(file, payload); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("read payload: %w", err)
	}

	category = payload[0]
	tokenCount = uint32((recordSize - 1) / 2)

	raw = make([]byte, 8+recordSize)
	copy(raw, sizeBuf[:])
	copy(raw[8:], payload)

	nextOffset = offset + 8 + int64(recordSize)
	return raw, category, tokenCount, nextOffset, nil
}
