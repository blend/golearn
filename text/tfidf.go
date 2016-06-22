package text

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/blendlabs/golearn/utilities"
	"github.com/gonum/matrix/mat64"
)

type TfidfVectorizer struct {
	Vocabulary []string
	VocabCount map[string]int
	IDF        *mat64.Dense
	Size       int
}

func NewTfidfVectorizer(vocabulary []string) *TfidfVectorizer {
	sort.Strings(vocabulary)
	vocabCount := make(map[string]int)
	for _, str := range vocabulary {
		vocabCount[str] = 0
	}

	return &TfidfVectorizer{
		vocabulary,
		vocabCount,
		nil,
		len(vocabulary),
	}
}

func (tfidf *TfidfVectorizer) Fit(data []string) error {
	if len(data) == 0 {
		return errors.New(fmt.Sprint("Cannot have 0 vocab words"))
	}

	for _, str := range tfidf.Vocabulary {
		tfidf.VocabCount[str] = 0
	}

	for _, textData := range data {
		for _, str := range strings.Split(textData, " ") {
			_, present := tfidf.VocabCount[str]
			if present {
				tfidf.VocabCount[str] += 1
			}
		}
	}
	idfArray := make([]float64, tfidf.Size)

	for index, str := range tfidf.Vocabulary {
		idfValue := 1 + math.Log(float64(tfidf.Size)/float64(tfidf.VocabCount[str]))
		idfArray[index] = idfValue
	}

	tfidf.IDF = mat64.NewDense(1, tfidf.Size, idfArray)

	return nil
}

func (tfidf *TfidfVectorizer) calculateTF(str string) []float64 {
	tfArray := make([]float64, tfidf.Size)
	strMap := make(map[string]int)
	for _, word := range strings.Split(str, " ") {
		strMap[word] = 1
	}

	for index, str := range tfidf.Vocabulary {
		_, present := strMap[str]
		if present {
			tfArray[index] = 1
		}
	}
	return tfArray
}

func (tfidf *TfidfVectorizer) Transform(data []string, matrix *mat64.Dense) {
	tfMatrix := mat64.NewDense(len(data), tfidf.Size, make([]float64, len(data)*tfidf.Size))
	idfMatrix := mat64.NewDense(len(data), tfidf.Size, make([]float64, len(data)*tfidf.Size))
	for index, str := range data {
		tfRow := tfidf.calculateTF(str)
		tfMatrix.SetRow(index, tfRow)
		idfMatrix.SetRow(index, tfidf.IDF.RawMatrix().Data)
	}

	matrix.MulElem(tfMatrix, idfMatrix)
}

func (tfidf *TfidfVectorizer) FitTransformAndSaveData(data []string, labels []string, matrix *mat64.Dense, saveDir string) error {
	tfidf.Fit(data)
	tfidf.Transform(data, matrix)
	return utilities.SaveMatrixDataToCSV(matrix, labels, saveDir)
}
